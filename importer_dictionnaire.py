#!/usr/bin/env python
"""
importer_dictionnaire.py : transforme le dictionnaire français -> kirundi
(data/dictionnaire_verbes_fr_rn.json) en données d'entraînement, et
cherche le PERFECTIF (forme du passé) de chaque verbe dans le livre FSI.

POURQUOI CHERCHER LE PERFECTIF ?
  Pour conjuguer au passé, il faut le perfectif : kora -> koze,
  genda -> giye. Il est irrégulier. Le script propose des formes
  possibles avec les modèles habituels du kirundi :
      -ra  -> -ye     (rindira -> rindiye, sigara -> sigaye)
      -ka  -> -tse    (garuka -> garutse, andika -> anditse)
      -ga  -> -ze     (vuga -> vuze)
      -nga -> -nze    (tanga -> tanze)
      -nda -> -nze    (kunda -> kunze)
      -na  -> -nye    (zana -> zanye)
      -ma  -> -mye    (rima -> rimye)
      -ba  -> -vye    (karaba -> karavye)
      -sha -> -shije  (fasha -> fashije)
      -za  -> -jije / -jeje  (baza -> bajije, heza -> hejeje)
      -ta  -> -se / -shije   (kubita -> kubise)
      ...
  puis il ne GARDE une forme que si elle apparaît vraiment dans le
  livre (par exemple "yarindiye" ou "barindiye"). Sinon le verbe va
  dans une liste "à compléter" pour un humain.

CE QUE LE SCRIPT PRODUIT
  1) data/verbes_rn.tsv : les verbes confirmés y sont AJOUTÉS
     (ceux qui y sont déjà ne sont pas touchés)
  2) data/verbes_a_completer.tsv : verbes sans perfectif confirmé,
     avec nos propositions. Remplis la colonne "perfectif" à la main,
     puis copie les lignes dans verbes_rn.tsv.
  3) --sortie : lexique_rn.txt (phase A) : « aimer » se dit gukunda...
  4) --sft : questions / réponses dans les deux sens (phase D)

Utilisation :
    python importer_dictionnaire.py \
        --dico data/dictionnaire_verbes_fr_rn.json \
        --mots /content/drive/MyDrive/donnees/fsi_rn/mots_frequents.tsv \
        --sortie /content/drive/MyDrive/donnees/lexique_rn \
        --sft /content/drive/MyDrive/donnees/qr/lexique_rn.jsonl
"""

import argparse
import json
import os
import random

VOYELLES = "aeiou"

# Ce qui peut se trouver DEVANT un perfectif dans le livre :
# nakoze, yakoze, bakoze, narakoze, ntiyakoze, nkoze, akoze...
DEVANTS = ["", "n", "u", "a", "tu", "du", "mu", "ba", "na", "wa", "ya", "twa",
           "mwa", "nara", "wara", "yara", "twara", "mwara", "bara", "sina",
           "ntiwa", "ntiya", "ntitwa", "ntimwa", "ntiba", "ki", "bi", "zi", "ca",
           "vya", "za", "ri", "rya", "ru", "rwa", "ka", "twa", "bwa", "bu", "ndi"]
# Même chose devant une voyelle : nanditse, yize, bize, ntiwize...
DEVANTS_V = ["", "n", "w", "y", "tw", "mw", "b", "nar", "war", "yar", "twar",
             "mwar", "bar", "sin", "ntiw", "ntiy", "ntitw", "ntimw", "ntib",
             "c", "vy", "z", "ry", "rw", "k", "bw", "nd", "ur", "ar", "tur", "mur"]


# Pièges vus en relisant : la forme est dans le livre, mais elle vient
# d'un AUTRE verbe. On la refuse.
FAUSSES_PREUVES = {
    "hiye": "guhya (mûrir, cuire), pas guha",
    "shimye": "gushima, pas gushimira",
}
# Verbes irréguliers qu'on ne conjugue pas avec les règles habituelles
# (ndafise = "j'ai" est une forme figée).
VERBES_SPECIAUX = {"kufita", "gufita", "kuba", "kutamenya", "kutubaha", "kutubahiriza"}

# Perfectifs corrigés à la main après relecture : la forme trouvée
# automatiquement venait d'un autre verbe (ex. "yishe" = il a tué, de
# kwica, pas de kwita). À vérifier par un locuteur.
CORRECTIONS = {
    "kwica": "ishe",       # yishe   (il a tué)
    "kwita": "ise",        # yise    (il a nommé)
    "kubura": "buze",      # yabuze  (il a manqué)
    "kwanka": "anse",      # yanse   (il a refusé)
    "kwakira": "akiriye",  # yakiriye (il a reçu) ; "yakize" vient de gukira
    "kuzura": "zutse",     # yazutse (il est ressuscité) ; "yuzuye" vient de kwuzura
    "kwumva": "umvise",    # yumvise (il a entendu)
}

# Verbes dont la forme trouvée est douteuse : on ne la garde pas, un
# humain doit trancher (forme dérivée ou d'un autre verbe).
DOUTEUX = {"kugara", "kwiyumvira", "gukuraho", "gutembera", "kunigana",
           "kuganira", "kuzeza", "kuduza"}


def racine_de(infinitif):
    """gukora -> kora ; kwandika -> andika ; kwumva -> umva"""
    for debut in ("kw", "gw"):
        if infinitif.startswith(debut) and infinitif[2:3] in VOYELLES:
            return infinitif[2:]
    for debut in ("ku", "gu"):
        if infinitif.startswith(debut):
            return infinitif[2:]
    return None


def propositions(r):
    """Formes possibles du perfectif, de la plus probable à la moins probable."""
    p = []
    if len(r) < 2:
        return p
    base = r[:-1]  # sans le "a" final
    if r.endswith("ra"):
        p += [r[:-2] + "ye", r[:-3] + "ye" if len(r) > 3 else None, r[:-2] + "ze"]
    if r.endswith("ka"):
        p += [r[:-2] + "tse", r[:-2] + "ste", r[:-2] + "kiye"]
    if r.endswith("nga"):
        p += [r[:-3] + "nze"]
    if r.endswith("ga"):
        p += [r[:-2] + "ze", r[:-2] + "giye"]
    if r.endswith("nda"):
        p += [r[:-3] + "nze"]
    if r.endswith("na"):
        p += [r[:-2] + "nye", r[:-2] + "niye"]
    if r.endswith("ma"):
        p += [r[:-2] + "mye", r[:-2] + "mvye"]
    if r.endswith("ba"):
        p += [r[:-2] + "vye", r[:-2] + "bye"]
    if r.endswith("sha"):
        p += [r[:-3] + "shije", r[:-3] + "sheje"]
    if r.endswith("za"):
        # PAS "-ze" : c'est la forme du SUBJONCTIF des verbes en -za
        # (ugerageze = "que tu essaies"), pas le passé.
        p += [r[:-2] + "jije", r[:-2] + "jeje", r[:-2] + "je"]
    if r.endswith("ta"):
        p += [r[:-2] + "se", r[:-2] + "she", r[:-2] + "shije", r[:-2] + "tse"]
    if r.endswith("ya"):
        p += [r[:-2] + "ye", r[:-2] + "ije"]
    if r.endswith("ha"):
        p += [r[:-2] + "shije", r[:-2] + "hiye"]
    # Formes "par défaut" seulement si aucun modèle précis ne s'applique
    # (sinon guhinduka donnerait "hindukiye", qui est le perfectif de
    # guhindukira : une fausse preuve).
    if not p:
        p += [base + "ye", base + "ije", base + "eje", base + "iye"]
    vus, sortie = set(), []
    for x in p:
        if x and len(x) >= 3 and x not in vus:
            vus.add(x)
            sortie.append(x)
    return sortie


def trouve_dans_livre(perf, mots, perfs_connus):
    """
    Le perfectif est confirmé si on le voit avec un préfixe connu.
    Règles strictes pour éviter les fausses preuves :
      - le perfectif d'un AUTRE verbe connu est refusé
        (ex. "giye" appartient à kugenda, pas à kugira) ;
      - une forme courte (moins de 6 lettres) doit être vue au moins
        2 fois avec des préfixes différents ;
      - une forme de moins de 4 lettres est toujours refusée.
    """
    if perf in perfs_connus or perf in FAUSSES_PREUVES or len(perf) < 4:
        return []
    devants = DEVANTS_V if perf[0] in VOYELLES else DEVANTS
    exemples = [d + perf for d in devants if (d + perf) in mots]
    # Un perfectif tout seul (sans préfixe) n'est pas une preuve suffisante
    # s'il est court : on demande au moins une forme avec préfixe.
    exemples = [e for e in exemples if e != perf]
    if len(perf) < 6 and len(exemples) < 2:
        return []
    return exemples


def lire_mots(chemin):
    mots = set()
    with open(chemin, encoding="utf-8") as f:
        for ligne in f:
            m = ligne.split("\t")[0].strip().strip(".?!,'").lower()
            if m:
                mots.add(m)
    return mots


def lire_tsv_existant(chemin):
    """Retourne (infinitifs connus, perfectifs connus)."""
    connus, perfs = set(), set()
    if os.path.exists(chemin):
        with open(chemin, encoding="utf-8") as f:
            for ligne in f:
                if ligne.strip() and not ligne.startswith("#"):
                    col = ligne.split("\t")
                    connus.add(col[0])
                    if len(col) > 2:
                        perfs.add(col[2])
    return connus, perfs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dico", default="data/dictionnaire_verbes_fr_rn.json")
    p.add_argument("--mots", required=True, help="mots_frequents.tsv de extraire_fsi.py")
    p.add_argument("--verbes", default="data/verbes_rn.tsv")
    p.add_argument("--a_completer", default="data/verbes_a_completer.tsv")
    p.add_argument("--sortie", required=True)
    p.add_argument("--sft")
    p.add_argument("--graine", type=int, default=42)
    args = p.parse_args()
    hasard = random.Random(args.graine)

    dico = json.load(open(args.dico, encoding="utf-8"))
    mots = lire_mots(args.mots)
    deja, perfs_connus = lire_tsv_existant(args.verbes)

    # --- 1) Regrouper : un verbe kirundi -> tous ses sens français -------
    sens = {}            # "gukora" -> ["faire", "travailler"]
    expressions = []     # (français, expression de plusieurs mots)
    vides = []
    for e in dico:
        fr = e["francais"].replace("’", "'")
        rn = [x.strip() for x in e.get("kirundi", []) if x and x.strip()]
        if not rn:
            vides.append(fr)
        for x in rn:
            if " " in x:
                expressions.append((fr, x))
            else:
                sens.setdefault(x, [])
                if fr not in sens[x]:
                    sens[x].append(fr)

    # --- 2) Chercher le perfectif de chaque verbe d'un seul mot -----------
    confirmes, a_completer = [], []
    for inf, frs in sens.items():
        if inf in deja:
            continue
        if inf in VERBES_SPECIAUX:
            a_completer.append((inf, "", "?", ", ".join(frs), "verbe irrégulier : à traiter à part"))
            continue
        r = racine_de(inf)
        if not r:
            a_completer.append((inf, "", "?", ", ".join(frs), "infinitif inhabituel"))
            continue
        if inf in CORRECTIONS:
            confirmes.append((inf, r, CORRECTIONS[inf], ", ".join(frs), "correction", ["corrigé à la main"]))
            continue
        if inf in DOUTEUX:
            props = " / ".join(propositions(r)[:3])
            a_completer.append((inf, r, "?", ", ".join(frs), f"forme trouvée douteuse ; propositions : {props}"))
            continue
        if r.endswith("nya") or (r.endswith("ya") and not r.endswith("aya")):
            # Pour ces verbes, le subjonctif et le passé peuvent s'écrire
            # pareil : la Bible ne permet pas de trancher.
            props = " / ".join(propositions(r)[:3])
            a_completer.append((inf, r, "?", ", ".join(frs), f"subjonctif et passé peuvent se ressembler ; propositions : {props}"))
            continue
        trouve = None
        for perf in propositions(r):
            ex = trouve_dans_livre(perf, mots, perfs_connus)
            if ex:
                trouve = (perf, ex)
                break
        if trouve:
            confirmes.append((inf, r, trouve[0], ", ".join(frs), "auto", trouve[1][:3]))
        else:
            props = " / ".join(propositions(r)[:3])
            a_completer.append((inf, r, "?", ", ".join(frs), f"propositions : {props}"))

    with open(args.verbes, "a", encoding="utf-8") as f:
        if confirmes:
            f.write("# --- ajoutés par importer_dictionnaire.py (perfectif vu dans le livre FSI ou la Bible) ---\n")
        for inf, r, perf, frs, src, _ in confirmes:
            f.write(f"{inf}\t{r}\t{perf}\t{frs}\t{src}\n")

    with open(args.a_completer, "w", encoding="utf-8") as f:
        f.write("# Verbes sans perfectif confirmé. Remplace \"?\" par le bon perfectif,\n")
        f.write("# puis copie la ligne (5 premières colonnes) dans data/verbes_rn.tsv.\n")
        f.write("# infinitif\tracine\tperfectif\tfrançais\tnote\n")
        for ligne in sorted(a_completer):
            f.write("\t".join(ligne) + "\n")

    # --- 3) Lexique pour la phase A et questions/réponses pour la phase D --
    par, qr = [], []
    for inf, frs in sens.items():
        for fr in frs:
            par.append(f"« {fr} » se dit {inf} en kirundi. {inf.capitalize()} veut dire « {fr} ».")
            qr.append({"user": f"Comment dit-on « {fr} » en kirundi ?", "assistant": f"On dit {inf}."})
        qr.append({"user": f"Que veut dire {inf} en français ?",
                   "assistant": f"{inf.capitalize()} veut dire « {' / '.join(frs)} »."})
    for fr, x in expressions:
        par.append(f"« {fr} » peut se dire « {x} » en kirundi.")
        qr.append({"user": f"Comment dit-on « {fr} » en kirundi ?", "assistant": f"On peut dire « {x} »."})

    hasard.shuffle(par)
    os.makedirs(args.sortie, exist_ok=True)
    with open(os.path.join(args.sortie, "lexique_rn.txt"), "w", encoding="utf-8") as f:
        f.write("\n\n".join(" ".join(par[i:i + 8]) for i in range(0, len(par), 8)) + "\n")
    if args.sft:
        os.makedirs(os.path.dirname(args.sft) or ".", exist_ok=True)
        with open(args.sft, "w", encoding="utf-8") as f:
            for q in qr:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")

    # --- 4) Rapport ---------------------------------------------------------
    print(f"Entrées du dictionnaire      : {len(dico)}")
    print(f"Verbes kirundi (un seul mot) : {len(sens)}  (déjà connus : {len(sens) - len(confirmes) - len(a_completer)})")
    print(f"Expressions de plusieurs mots: {len(expressions)}")
    print(f"Sans traduction              : {len(vides)} ({', '.join(vides)})")
    print(f"\nPerfectif CONFIRMÉ dans le livre : {len(confirmes)}")
    for inf, r, perf, frs, _, ex in confirmes:
        print(f"  {inf:16s} {perf:14s} vu : {', '.join(ex)}")
    print(f"\nÀ compléter à la main : {len(a_completer)}  -> {args.a_completer}")
    print(f"Lexique : {len(par)} phrases ; questions/réponses : {len(qr)}")


if __name__ == "__main__":
    main()
