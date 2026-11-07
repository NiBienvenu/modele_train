#!/usr/bin/env python
"""
gen_conjugaison_rn.py : crée automatiquement des conjugaisons en kirundi
pour la phase A.

D'où viennent les règles ?
  Du cours FSI "Kirundi Basic Course" (domaine public), vérifiées sur les
  formes réelles du livre (voir extraire_fsi.py). Les formes du passé
  (perfectif) sont irrégulières : elles sont écrites à la main dans
  data/verbes_rn.tsv, on ne les devine pas.

COMMENT UN VERBE KIRUNDI EST CONSTRUIT
  [négation] + SUJET + [TEMPS] + RACINE (ou PERFECTIF)

  Sujets :  n- (jewe)  u- (wewe)  a- (we)  tu- (twebwe)  mu- (mwebwe)  ba- (bo)

  Temps produits ici :
    présent (-ra-)     : nda-kora, ura-kora, ara-kora ...
    présent (sans -ra-): n-kora, u-kora ... (avec un complément : "Nkora akazi")
    futur (-zo-)       : nzo-kora, uzo-kora ...
    passé (-a-)        : na-koze, wa-koze, ya-koze ...
    passé (-a-ra-)     : nara-koze, wara-koze ...
    parfait            : n-koze, u-koze ... ("je viens de faire")
    négatif présent    : sin-kora, ntu-kora, nta-kora, ntidu-kora ...
    négatif passé      : sina-koze, ntiwa-koze ...
    négatif futur      : sinzo-kora, ntuzo-kora ...

  Remarque du livre FSI : le passé d'aujourd'hui et le passé d'avant
  aujourd'hui ne se distinguent que par le TON, qui ne s'écrit pas.
  À l'écrit, "nakoze" sert donc pour les deux.

LES RÈGLES DE SONS (toutes vues dans le livre)
  1) n + consonne : n+b -> mb (mbona), n+f -> mf (mfasha), n+h -> mp (mpa),
     n+r -> nd (ndya), n+k -> nk (nkora), n+s -> ns (nsoma)...
  2) Devant une voyelle : u -> w, a -> y, tu -> tw, mu -> mw, ba -> b
     (wandika, yandika, twandika, mwandika, bandika).
  3) Loi de Dahl : tu -> du devant une consonne sourde (p t k c f s h)
     (dukora, mais tugenda ; gukora, mais kugenda).
  4) -ra- + voyelle : ndandika, ndiga, ndumva (le "a" de -ra- tombe).
  5) Futur d'un verbe à voyelle : zo + kw + racine (nzokwigisha).

Utilisation dans Colab :
    !python gen_conjugaison_rn.py \
        --verbes data/verbes_rn.tsv \
        --sortie /content/drive/MyDrive/donnees/conjugaison_rn \
        --sft /content/drive/MyDrive/donnees/qr/conjugaison_rn.jsonl \
        --verifier /content/drive/MyDrive/donnees/fsi_rn/mots_frequents.tsv
"""

import argparse
import json

from separation import verbe_de_test
import os
import random

VOYELLES = "aeiou"
SOURDES = ("p", "t", "k", "c", "f", "s", "h")  # consonnes "sourdes" (loi de Dahl)

# (pronom, préfixe sujet, nom de la personne en français)
PERSONNES = [
    ("jewe", "n", "je"),
    ("wewe", "u", "tu"),
    ("we", "a", "il/elle"),
    ("twebwe", "tu", "nous"),
    ("mwebwe", "mu", "vous"),
    ("bo", "ba", "ils/elles"),
]


# ---------------------------------------------------------------------------
# Les règles de sons
# ---------------------------------------------------------------------------

def n_plus(suite):
    """
    Règle 1 : le préfixe "n" (je) change selon le son qui suit.
      n + bona -> mbona     n + hagarara -> mpagarara
      n + rya  -> ndya      n + andika   -> nandika
    """
    if not suite or suite[0] in VOYELLES:
        return "n" + suite
    c = suite[0]
    if c in "bvfp":
        return "m" + suite
    if c == "h":
        return "mp" + suite[1:]
    if c == "r":
        return "nd" + suite[1:]
    if c in "mn":       # n + menya -> menya, n + nywa -> nywa
        return suite
    return "n" + suite  # k, g, t, d, s, sh, z, c, j...


def sujet(prefixe, suite):
    """
    Colle le préfixe sujet devant la suite (marque de temps ou racine)
    en appliquant les règles 1, 2 et 3.
    """
    if prefixe == "n":
        return n_plus(suite)
    if suite and suite[0] in VOYELLES:
        # Règle 2 : u -> w, a -> y, tu -> tw, mu -> mw, ba -> b
        return {"u": "w", "a": "y", "tu": "tw", "mu": "mw", "ba": "b"}[prefixe] + suite
    if prefixe == "tu" and suite.startswith(SOURDES):
        return "du" + suite  # Règle 3 : loi de Dahl
    return prefixe + suite


def avec_ra(suite):
    """Règle 4 : ra + andika -> randika, ra + kora -> rakora."""
    return "r" + suite if suite[0] in VOYELLES else "ra" + suite


def avec_a_passe(prefixe, perfectif):
    """
    Passé : sujet + a + perfectif.
      na-koze, wa-koze, ya-koze, twa-koze, mwa-koze, ba-koze (ba + a = ba)
    Si le perfectif commence par une voyelle, le "a" se fond dedans :
      n + anditse -> nanditse, mw + ize -> mwize, n + umvise -> numvise
    """
    if perfectif[0] in VOYELLES:
        return sujet(prefixe, perfectif)
    return {"n": "na", "u": "wa", "a": "ya", "tu": "twa",
            "mu": "mwa", "ba": "ba"}[prefixe] + perfectif


def avec_zo(racine):
    """
    Règle 5 : zo + kora -> zokora ; zo + iga -> zokwiga ; zo + ubaka -> zokwubaka
    (Cox, Livre I leçon 36 : bazokwubaka, ntazokwubaka).
    Le futur -zo- ne sert que pour APRÈS aujourd'hui (Cox leçon 29) :
    pour aujourd'hui on dit "Ndagenda uyu musi".
    """
    if racine[0] in VOYELLES:
        return "zokw" + racine
    return "zo" + racine


def subjonctif(racine):
    """Racine terminée en -e : kora -> kore ; andika -> andike (Cox Livre II)."""
    return racine[:-1] + "e" if racine.endswith("a") else racine


def avec_o(racine):
    """
    Conditionnel (Cox leçon 111) : sujet + o + racine.
    nokunda, wokunda, yokunda, twokunda, mwokunda, bokunda ;
    verbe à voyelle : o + kw + racine (mwokwemera).
    """
    return "okw" + racine if racine[0] in VOYELLES else "o" + racine


def negatif(prefixe, suite):
    """
    Négation : si- pour "je", nti- pour les autres.
      sin-kora, ntu-kora, nta-kora, ntidu-kora, ntimu-kora, ntiba-kora
    """
    if prefixe == "n":
        return "si" + n_plus(suite)
    forme = sujet(prefixe, suite)
    if prefixe == "u" and not forme.startswith("w"):
        return "nt" + forme       # nti + u -> ntu
    if prefixe == "a" and not forme.startswith("y"):
        return "nt" + forme       # nti + a -> nta
    return "nti" + forme


# ---------------------------------------------------------------------------
# Les temps
# ---------------------------------------------------------------------------

def conjuguer(racine, perfectif):
    """
    Retourne {nom du temps : [6 formes]} (None = forme non produite).
    Le nom du temps dit aussi QUAND l'employer, pour que le modèle
    apprenne le bon sens en français (Cox, Livre I leçons 29, 71, 75, 111, 122).
    """
    zo = avec_zo(racine)
    sub = subjonctif(racine)
    t = {}
    t["présent"] = [sujet(p, avec_ra(racine)) for _, p, _ in PERSONNES]
    # Sans -ra- : avec un complément ("Nkora akazi") ou en subordonnée
    t["présent sans -ra- (avec complément)"] = [sujet(p, racine) for _, p, _ in PERSONNES]
    t["futur (après aujourd'hui)"] = [sujet(p, zo) for _, p, _ in PERSONNES]
    # nakoze : aujourd'hui OU avant (seul le ton change, il ne s'écrit pas)
    t["passé"] = [avec_a_passe(p, perfectif) for _, p, _ in PERSONNES]
    # narakoze : hier ou avant, et SEULEMENT si rien ne suit le verbe (leçon 71)
    t["passé lointain avec -ra- (sans complément)"] = [
        {"n": "na", "u": "wa", "a": "ya", "tu": "twa", "mu": "mwa", "ba": "ba"}[p]
        + avec_ra(perfectif) for _, p, _ in PERSONNES]
    # ndakoze : "je viens de travailler" (leçon 75) ; aussi l'état présent :
    # ndarwaye = je suis malade, biragoye = c'est difficile
    t["passé immédiat (je viens de)"] = [sujet(p, avec_ra(perfectif)) for _, p, _ in PERSONNES]
    t["négatif présent"] = [negatif(p, racine) for _, p, _ in PERSONNES]
    t["négatif passé"] = [
        negatif(p, perfectif) if perfectif[0] in VOYELLES
        else negatif(p, "a" + perfectif) for _, p, _ in PERSONNES]
    t["négatif futur"] = [negatif(p, zo) for _, p, _ in PERSONNES]
    t["conditionnel"] = [sujet(p, avec_o(racine)) for _, p, _ in PERSONNES]
    t["négatif conditionnel"] = [negatif(p, avec_o(racine)) for _, p, _ in PERSONNES]
    t["subjonctif (que je ...)"] = [sujet(p, sub) for _, p, _ in PERSONNES]
    # Impératif : tu = racine seule (kora), vous = mu + subjonctif (mukore)
    imp = [None] * 6
    imp[1], imp[4] = racine, sujet("mu", sub)
    t["impératif"] = imp
    imp_neg = [None] * 6
    imp_neg[1], imp_neg[4] = negatif("u", sub), negatif("mu", sub)
    t["impératif négatif"] = imp_neg
    return t


# Petits mots de temps pour mettre les verbes en situation.
MOTS_TEMPS = {"présent": "Ubu", "futur (après aujourd'hui)": "Ejo", "passé": "Ejo",
              "négatif futur": "Ejo", "négatif passé": "Ejo"}


# ---------------------------------------------------------------------------
# Lecture de la liste des verbes
# ---------------------------------------------------------------------------

def lire_verbes(chemin):
    verbes = []
    with open(chemin, encoding="utf-8") as f:
        for ligne in f:
            if not ligne.strip() or ligne.startswith("#"):
                continue
            col = ligne.rstrip("\n").split("\t")
            if len(col) < 3 or not col[1] or not col[2] or col[2] == "?":
                print(f"  ligne ignorée : {ligne.strip()[:60]}")
                continue
            # Le sens français peut être vide (verbes de Cox pas encore traduits) :
            # on produit alors seulement du kirundi.
            verbes.append({"inf": col[0], "racine": col[1], "perf": col[2],
                           "fr": col[3].strip() if len(col) > 3 else ""})
    return verbes


# ---------------------------------------------------------------------------
# Textes d'entraînement
# ---------------------------------------------------------------------------

def textes(v, hasard):
    tout = conjuguer(v["racine"], v["perf"])
    par, qr = [], []
    for nom, formes in tout.items():
        ok = [f for f in formes if f]
        if not ok:
            continue
        liste = ", ".join(ok)
        # Bilingue : le modèle relie le verbe kirundi à son sens français.
        if v["fr"]:
            par.append(f"Le verbe {v['inf']} ({v['fr']}) au {nom} : {liste}.")
        # Kirundi seul : avec les pronoms.
        par.append(" ".join(f"{pr.capitalize()} {f}."
                            for (pr, _, _), f in zip(PERSONNES, formes) if f))
        qr.append({"user": f"Conjugue le verbe {v['inf']} au {nom}.",
                   "assistant": f"{liste}."})

    # Petites phrases avec un mot de temps
    phrases = []
    for nom, mot in MOTS_TEMPS.items():
        formes = [f for f in tout[nom] if f]
        if formes:
            phrases.append(f"{mot} {hasard.choice(formes)}.")
    par.append(" ".join(phrases))

    if v["fr"]:
        qr.append({"user": f"Que veut dire le verbe {v['inf']} ?",
                   "assistant": f"{v['inf'].capitalize()} veut dire « {v['fr']} »."})
        qr.append({"user": f"Comment dit-on « {v['fr'].split(',')[0]} » en kirundi ?",
                   "assistant": f"On dit {v['inf']}."})
    return par, qr, tout


def verifier(tout_formes, chemin_mots):
    """
    Compare nos formes avec les mots trouvés dans le livre FSI
    (mots_frequents.tsv, créé par extraire_fsi.py). Une forme trouvée dans
    le livre est sûre. Une forme absente n'est PAS forcément fausse : le
    livre ne contient pas toutes les formes de tous les verbes.
    """
    livre = set()
    with open(chemin_mots, encoding="utf-8") as f:
        for ligne in f:
            mot = ligne.split("\t")[0].strip().strip(".?!,").lower()
            if mot:
                livre.add(mot)
    par_temps = {}
    for formes in tout_formes:
        for nom, liste in formes.items():
            for f in liste:
                if f:
                    n_ok, n = par_temps.get(nom, (0, 0))
                    par_temps[nom] = (n_ok + (f in livre), n + 1)
    print("\nFormes retrouvées telles quelles dans le livre FSI :")
    for nom, (n_ok, n) in par_temps.items():
        print(f"  {nom:20s} {n_ok:4d} / {n}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--verbes", action="append",
                   help="fichier(s) de verbes ; par défaut data/verbes_rn.tsv et data/verbes_cox.tsv")
    p.add_argument("--sortie", required=True)
    p.add_argument("--sft")
    p.add_argument("--repeter", type=int, default=5,
                   help="chaque verbe est produit N fois (phrases différentes)")
    p.add_argument("--verifier", help="mots_frequents.tsv de extraire_fsi.py")
    p.add_argument("--inclure_tests", action="store_true",
                   help="ne PAS exclure les verbes de test (seulement pour fabriquer les tests)")
    p.add_argument("--graine", type=int, default=42)
    args = p.parse_args()

    hasard = random.Random(args.graine)
    fichiers = args.verbes or [f for f in ("data/verbes_rn.tsv", "data/verbes_cox.tsv") if os.path.exists(f)]
    verbes, vus, exclus = [], set(), []
    for fichier in fichiers:
        for v in lire_verbes(fichier):
            if v["inf"] in vus:
                continue
            if verbe_de_test(v["inf"]) and not args.inclure_tests:
                exclus.append(v["inf"])   # gardé pour les tests : jamais vu à l'entraînement
                continue
            vus.add(v["inf"])             # le premier fichier a priorité
            verbes.append(v)
    print(f"{len(verbes)} verbes lus ({', '.join(fichiers)}) ; {len(exclus)} gardés pour les tests")

    tous_par, tous_qr, tout_formes = [], [], []
    for k in range(args.repeter):
        for v in verbes:
            # Les verbes de Cox (sans sens français) sont nombreux : 2 passages suffisent
            if k >= 2 and not v["fr"]:
                continue
            par, qr, formes = textes(v, hasard)
            # Les tableaux sont identiques à chaque répétition : on ne les
            # garde qu'une fois. Les petites phrases, elles, changent.
            tous_par.extend(par if k == 0 else par[-1:])
            if k == 0:
                tous_qr.extend(qr)
                tout_formes.append(formes)

    hasard.shuffle(tous_par)
    os.makedirs(args.sortie, exist_ok=True)
    chemin = os.path.join(args.sortie, "conjugaison_rn_001.txt")
    with open(chemin, "w", encoding="utf-8") as f:
        f.write("\n\n".join(tous_par) + "\n")
    print(f"{len(tous_par)} paragraphes écrits dans {chemin}")

    if args.sft:
        os.makedirs(os.path.dirname(args.sft) or ".", exist_ok=True)
        with open(args.sft, "w", encoding="utf-8") as f:
            for q in tous_qr:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")
        print(f"{len(tous_qr)} questions/réponses écrites dans {args.sft}")

    if args.verifier:
        verifier(tout_formes, args.verifier)

    # Un tableau complet à relire à l'oeil
    print("\nExemple à vérifier :")
    for v in verbes[:1] + [x for x in verbes if x["racine"][0] in VOYELLES][:1]:
        for nom, formes in conjuguer(v["racine"], v["perf"]).items():
            print(f"  {v['inf']:10s} {nom:20s} {', '.join(f or '-' for f in formes)}")


if __name__ == "__main__":
    main()
