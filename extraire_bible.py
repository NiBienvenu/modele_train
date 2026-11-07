#!/usr/bin/env python
"""
extraire_bible.py : extrait la Bible en kirundi et en français, et les
ALIGNE verset par verset.

POURQUOI ?
  La Bible est le plus grand texte qui existe dans les deux langues, avec
  la même phrase de chaque côté. Le modèle y voit les mots kirundi en
  contexte : tous les temps, les accords, les expressions. On ne
  s'intéresse PAS au sens religieux, seulement à la LANGUE.

LE KIRUNDI (PDF mis en page) : on lit la TAILLE des caractères
  taille 10, normal   -> texte du verset            : gardé
  taille 6,5, chiffre -> numéro de verset (1, 4a)   : sert à aligner
  taille 6,5, lettre  -> appel de note (a, b, c)    : SUPPRIMÉ
  taille 10, gras     -> titre de livre, numéro de chapitre, intertitre
  taille < 9          -> notes en bas de page       : SUPPRIMÉES

LE FRANÇAIS (Bible de Jérusalem) : chaque verset commence par
  "Genèse 6, 8 ..." ; les lignes suivantes sont la suite du verset.

LES DEUX BIBLES ont 73 livres dans le même ordre : le livre n°1 kirundi
correspond au livre n°1 français, etc. Un verset est aligné s'il existe
des deux côtés avec le même (livre, chapitre, verset).

CE QUE LE SCRIPT PRODUIT (dans --sortie)
  bible_rn.txt            : kirundi seul (un paragraphe par chapitre)
  bible_fr.txt            : français seul
  bible_paires.jsonl      : {"ref", "fr", "rn"} pour chaque verset aligné
  bible_bilingue.txt      : paires mélangées pour la phase A
                            ("Kirundi : ... / Français : ...")
  mots_bible_rn.tsv       : TOUS les mots kirundi, nombre, exemple (référence)
  mots_bible_fr.tsv       : pareil pour le français
  rapport.txt             : chiffres par livre, pour vérifier
  (--sft) questions/réponses de traduction pour la phase D

Utilisation dans Colab :
    !pip install pymupdf
    !python extraire_bible.py \
        --rn /content/drive/MyDrive/sources/bible_kirundi.pdf \
        --fr /content/drive/MyDrive/sources/bible_francais.pdf \
        --sortie /content/drive/MyDrive/donnees/bible \
        --sft /content/drive/MyDrive/donnees/qr/traduction_bible.jsonl
"""

import argparse
import collections
import json
import os
import random
import re
import unicodedata

import pymupdf

# Titres qui ouvrent un NOUVEAU LIVRE dans la Bible kirundi
DEBUT_LIVRE_RN = re.compile(r"^(IGITABU |INKURU NZIZA |IKETE )")
# Titre de psaume : "IZABURI YA 23", "IZABURI Y'7"
PSAUME_RN = re.compile(r"^IZABURI Y(?:A |')(\d+)\b")
NUM_VERSET = re.compile(r"^(\d+)([a-z]?)$")

# Les 73 livres, dans l'ordre (le même dans les deux Bibles)
NOMS_FR = [
    "Genèse", "Exode", "Lévitique", "Nombres", "Deutéronome", "Josué", "Juges",
    "Ruth", "1 Samuel", "2 Samuel", "1 Rois", "2 Rois", "1 Chroniques",
    "2 Chroniques", "Esdras", "Néhémie", "Tobie", "Judith", "Esther",
    "1 Maccabées", "2 Maccabées", "Job", "Psaumes", "Proverbes", "Ecclésiaste",
    "Cantique", "Sagesse", "Ecclésiastique", "Isaïe", "Jérémie", "Lamentations",
    "Baruch", "Ezéchiel", "Daniel", "Osée", "Joël", "Amos", "Abdias", "Jonas",
    "Michée", "Nahum", "Habaquq", "Sophonie", "Aggée", "Zacharie", "Malachie",
    "Matthieu", "Marc", "Luc", "Jean", "Actes", "Romains", "1 Corinthiens",
    "2 Corinthiens", "Galates", "Ephésiens", "Philippiens", "Colossiens",
    "1 Thessaloniciens", "2 Thessaloniciens", "1 Timothée", "2 Timothée",
    "Tite", "Philémon", "Hébreux", "Jacques", "1 Pierre", "2 Pierre", "1 Jean",
    "2 Jean", "3 Jean", "Jude", "Apocalypse",
]
# Livres d'un seul chapitre : le chapitre est mis à 1 des deux côtés
UN_CHAPITRE = {"Abdias", "Philémon", "2 Jean", "3 Jean", "Jude"}

# Étiquettes vraiment écrites dans le PDF français (avec leurs fautes :
# "Macchabées", "Théssaloniciens", "Thimothées", "Psaume"...). Joël y est
# étiqueté "Amos" par erreur : l'ordre des livres corrige cela.
ETIQUETTES_FR = sorted({
    "Genèse", "Exode", "Lévitique", "Nombres", "Deutéronome", "Josué", "Juges",
    "Ruth", "1 Samuel", "2 Samuel", "1 Rois", "2 Rois", "1 Chroniques",
    "2 Chroniques", "Esdras", "Néhémie", "Tobie", "Judith", "Esther",
    "1 Macchabées", "2 Maccabées", "Job", "Psaume", "Proverbes", "Ecclésiaste",
    "Cantique", "Sagesse", "Ecclésiastique", "Isaïe", "Jérémie", "Lamentations",
    "Baruch", "Ezéchiel", "Daniel", "Osée", "Joël", "Amos", "Abdias", "Jonas",
    "Michée", "Nahum", "Habaquq", "Sophonie", "Aggée", "Zacharie", "Malachie",
    "Matthieu", "Marc", "Luc", "Jean", "Actes", "Romains", "1 Corinthiens",
    "2 Corinthiens", "Galates", "Ephésiens", "Philippiens", "Colossiens",
    "1 Théssaloniciens", "2 Théssaloniciens", "1 Thimothées", "2 Thimothées",
    "Tite", "Philémon", "Hébreux", "Jacques", "1 Pierre", "2 Pierre", "1 Jean",
    "2 Jean", "3 Jean", "Jude", "Apocalypse"}, key=len, reverse=True)
# Un début de verset, n'importe où dans la ligne : "Genèse 6, 8 " ou "Jude 9 "
VERSET_FR = re.compile(
    r"(?:^|(?<=[\s.,;:!?\"»)]))(" + "|".join(re.escape(e) for e in ETIQUETTES_FR)
    + r")\s+(?:(\d+),\s*)?(\d+)\s")


def sans_tons(texte):
    """wărāra -> warara (les tons ne s'écrivent pas en kirundi courant)."""
    d = unicodedata.normalize("NFD", texte)
    d = "".join(c for c in d if c not in "̆̄́̀̂̌")
    return unicodedata.normalize("NFC", d)


def propre(texte):
    texte = texte.replace("­", "").replace("\t", " ")
    return re.sub(r"\s+", " ", texte).strip()


# ---------------------------------------------------------------------------
# 1. Kirundi
# ---------------------------------------------------------------------------

def lire_kirundi(chemin, garder_tons=False):
    """
    Retourne :
      versets : {(n_livre, chapitre, verset): texte}
      titres  : liste des titres de livres (dans l'ordre)
    """
    doc = pymupdf.open(chemin)
    versets = collections.OrderedDict()
    titres = []
    livre, chap, vers = 0, 0, 0
    morceaux = collections.defaultdict(list)

    def ajouter(t):
        if livre and chap and vers:
            morceaux[(livre, chap, vers)].append(t)

    for page in doc:
        for bloc in page.get_text("dict")["blocks"]:
            for ligne in bloc.get("lines", []):
                spans = ligne["spans"]
                texte_gras = "".join(s["text"] for s in spans
                                     if "Bold" in s["font"] and s["size"] >= 9.5).strip()
                # --- Titre de livre ---
                if texte_gras and DEBUT_LIVRE_RN.match(texte_gras) and texte_gras.upper() == texte_gras:
                    if not titres or titres[-1] != texte_gras:
                        titres.append(texte_gras)
                        livre, chap, vers = len(titres), 0, 0
                    continue
                # --- Psaume : le numéro du psaume = le chapitre ---
                m = PSAUME_RN.match(texte_gras)
                if m:
                    chap, vers = int(m.group(1)), 0
                    continue
                for s in spans:
                    t, taille, gras = s["text"], s["size"], "Bold" in s["font"]
                    if not t.strip():
                        if 9.5 <= taille <= 10.5:
                            ajouter(" ")
                        continue
                    if gras and 9.5 <= taille:
                        # Numéro de chapitre (gras, seul) ; sinon intertitre : ignoré
                        if t.strip().isdigit():
                            chap, vers = int(t.strip()), 0
                        continue
                    if 6 <= taille < 7.5:
                        m = NUM_VERSET.match(t.strip())
                        if m:
                            vers = int(m.group(1))   # "4a" et "4b" -> verset 4
                            ajouter(" ")
                        # sinon : appel de note (a, b, c...) -> supprimé
                        continue
                    if taille < 9:
                        continue                     # notes en bas de page
                    if 9.5 <= taille <= 10.5:
                        ajouter(t)
                # Fin de ligne = espace (le texte continue à la ligne suivante)
                ajouter(" ")

    for cle, liste in morceaux.items():
        t = propre("".join(liste))
        l, c, v = cle
        if l <= len(NOMS_FR) and NOMS_FR[l - 1] in UN_CHAPITRE:
            cle = (l, 1, v)
        if not garder_tons:
            t = sans_tons(t)
        if t:
            versets[cle] = (versets[cle] + " " + t) if cle in versets else t
    return versets, titres


# ---------------------------------------------------------------------------
# 2. Français
# ---------------------------------------------------------------------------

VERSET_SANS_NOM = re.compile(r"^(?:(\d+),\s*)?(\d+)\s+(?=\S)")


def lire_francais(chemin):
    """
    Retourne {(n_livre, chapitre, verset): texte}.

    Étape 1 : on lit tous les débuts de verset : "Genèse 6, 8 ...".
      Au début de chaque livre, le PDF écrit le titre seul sur une ligne,
      puis "1, 1 ..." SANS le nom : on garde aussi ces versets-là.
    Étape 2 : on découpe en livres. On ne se fie pas au nom écrit (Joël
      est étiqueté "Amos" par erreur) : un nouveau livre commence quand
      le nom change, ou quand la numérotation repart en arrière.
      Les 73 livres sont dans le même ordre que la Bible kirundi.
    """
    doc = pymupdf.open(chemin)
    etiquettes = set(ETIQUETTES_FR)
    recs = []              # [étiquette ou None, chapitre, verset, [morceaux]]
    apres_titre = False
    for page in doc:
        texte = unicodedata.normalize("NFC", page.get_text("text"))
        for ligne in texte.splitlines():
            brut = ligne.strip(" \r\xa0")
            if not brut:
                continue
            # Ligne de titre de livre : "Lévitique<TAB>"
            if "\t" in ligne and ligne.split("\t")[0].strip() in etiquettes \
                    and not ligne.split("\t", 1)[1].strip(" \r\xa0"):
                apres_titre = True
                continue
            if apres_titre:
                m = VERSET_SANS_NOM.match(brut)
                apres_titre = False
                if m:
                    chap = int(m.group(1)) if m.group(1) else 1
                    recs.append([None, chap, int(m.group(2)), [brut[m.end():]]])
                    continue
            pos = 0
            for m in VERSET_FR.finditer(ligne):
                if recs:
                    recs[-1][3].append(ligne[pos:m.start()])
                chap = int(m.group(2)) if m.group(2) else 1
                recs.append([m.group(1), chap, int(m.group(3)), []])
                pos = m.end()
            if recs:
                recs[-1][3].append(" " + ligne[pos:])

    # On ignore la table des matières : le texte commence à "Genèse 1, 1"
    for i, r in enumerate(recs):
        if r[0] == "Genèse" and r[1] == 1 and r[2] == 1:
            recs = recs[i:]
            break

    # Étape 2 : découpage en livres
    versets = collections.OrderedDict()
    livre, etiquette, dernier, sans_nom = 0, None, (0, 0), False
    for eti, chap, vers, morceaux in recs:
        if eti is None:
            if (chap, vers) > (1, 1):
                # Ligne qui commence par un chiffre après un titre répété
                # (ex. "Colossiens" écrit deux fois) : pas un début de livre
                if versets:
                    k = next(reversed(versets))
                    versets[k] += " " + "".join(morceaux)
                continue
            # Premier verset d'un livre, juste après son titre
            livre += 1
            etiquette, sans_nom = None, True
        elif eti != etiquette:
            if sans_nom:
                etiquette, sans_nom = eti, False      # même livre, on apprend son nom
            elif (chap, vers) <= (1, 3) or chap == 0:
                livre += 1
                etiquette = eti
            else:
                # Nom de livre cité au milieu du texte : pas un verset
                if versets:
                    k = next(reversed(versets))
                    versets[k] += f" {eti} {chap}, {vers} " + "".join(morceaux)
                continue
        elif (chap, vers) <= (1, 3) and dernier[0] >= 2:
            # La numérotation repart à zéro alors qu'on était déjà loin
            # (chapitre 2 ou plus) : nouveau livre mal étiqueté (Joël = "Amos").
            # Un retour à 1,1 depuis le chapitre 1 (prologue de
            # l'Ecclésiastique mal numéroté) ne compte PAS.
            livre += 1
        dernier = (chap, vers)
        if livre == 0 or livre > len(NOMS_FR):
            continue
        if NOMS_FR[livre - 1] in UN_CHAPITRE or chap == 0:
            chap = max(chap, 1) if NOMS_FR[livre - 1] not in UN_CHAPITRE else 1
        cle = (livre, chap, vers)
        t = "".join(morceaux)
        versets[cle] = (versets[cle] + " " + t) if cle in versets else t
    for k in versets:
        versets[k] = propre(versets[k])
    return versets, NOMS_FR[:livre]


# ---------------------------------------------------------------------------
# 3. Mots
# ---------------------------------------------------------------------------

MOT = re.compile(r"[A-Za-zÀ-ÿ]+(?:'[A-Za-zÀ-ÿ]+)*")


def compter_mots(versets, refs):
    compte, exemple = collections.Counter(), {}
    for cle, texte in versets.items():
        for m in MOT.findall(texte):
            mot = m.lower()
            compte[mot] += 1
            if mot not in exemple:
                exemple[mot] = refs(cle)
    return compte, exemple


# ---------------------------------------------------------------------------
# 4. Programme principal
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rn", required=True)
    p.add_argument("--fr", required=True)
    p.add_argument("--sortie", required=True)
    p.add_argument("--sft")
    p.add_argument("--max_sft", type=int, default=30000)
    p.add_argument("--garder_tons", action="store_true")
    p.add_argument("--graine", type=int, default=42)
    args = p.parse_args()
    hasard = random.Random(args.graine)
    os.makedirs(args.sortie, exist_ok=True)

    print("Lecture du kirundi...")
    rn, titres = lire_kirundi(args.rn, args.garder_tons)
    print(f"  {len(titres)} livres, {len(rn)} versets")
    print("Lecture du français...")
    fr, noms = lire_francais(args.fr)
    print(f"  {len(noms)} livres, {len(fr)} versets")
    if len(titres) != len(noms):
        print(f"ATTENTION : {len(titres)} livres kirundi mais {len(noms)} livres français.")
        print("  Les livres sont alignés par leur numéro d'ordre : vérifie rapport.txt.")

    def ref(cle):
        l, c, v = cle
        nom = noms[l - 1] if l <= len(noms) else f"livre {l}"
        return f"{nom} {c},{v}"

    # --- Alignement ----------------------------------------------------------
    paires = []
    par_livre = collections.defaultdict(lambda: [0, 0, 0])
    for cle, t_rn in rn.items():
        par_livre[cle[0]][0] += 1
        if cle in fr:
            t_fr = fr[cle]
            par_livre[cle[0]][1] += 1
            # Contrôle : si les longueurs sont trop différentes, les versets
            # ne disent sans doute pas la même chose (numérotation différente).
            ratio = len(t_rn) / max(len(t_fr), 1)
            if 0.5 <= ratio <= 2.5:
                par_livre[cle[0]][2] += 1
                paires.append({"ref": ref(cle), "fr": t_fr, "rn": t_rn})

    # --- Fichiers ------------------------------------------------------------
    def par_chapitre(versets):
        chap = collections.OrderedDict()
        for (l, c, v), t in versets.items():
            chap.setdefault((l, c), []).append(t)
        return ["\n".join(ts) for ts in chap.values()]

    with open(os.path.join(args.sortie, "bible_rn.txt"), "w", encoding="utf-8") as f:
        f.write("\n\n".join(par_chapitre(rn)) + "\n")
    with open(os.path.join(args.sortie, "bible_fr.txt"), "w", encoding="utf-8") as f:
        f.write("\n\n".join(par_chapitre(fr)) + "\n")
    with open(os.path.join(args.sortie, "bible_paires.jsonl"), "w", encoding="utf-8") as f:
        for pr in paires:
            f.write(json.dumps(pr, ensure_ascii=False) + "\n")

    # Texte bilingue pour la phase A : on alterne le sens (rn->fr, fr->rn)
    melange = paires[:]
    hasard.shuffle(melange)
    blocs = []
    for i in range(0, len(melange), 4):
        lignes = []
        for pr in melange[i:i + 4]:
            if hasard.random() < 0.5:
                lignes.append(f"Kirundi : {pr['rn']}\nFrançais : {pr['fr']}")
            else:
                lignes.append(f"Français : {pr['fr']}\nKirundi : {pr['rn']}")
        blocs.append("\n".join(lignes))
    with open(os.path.join(args.sortie, "bible_bilingue.txt"), "w", encoding="utf-8") as f:
        f.write("\n\n".join(blocs) + "\n")

    if args.sft:
        os.makedirs(os.path.dirname(args.sft) or ".", exist_ok=True)
        with open(args.sft, "w", encoding="utf-8") as f:
            for pr in melange[: args.max_sft]:
                if hasard.random() < 0.5:
                    q = {"user": f"Traduis en français : {pr['rn']}", "assistant": pr["fr"]}
                else:
                    q = {"user": f"Traduis en kirundi : {pr['fr']}", "assistant": pr["rn"]}
                f.write(json.dumps(q, ensure_ascii=False) + "\n")

    # --- Liste de tous les mots ---------------------------------------------
    for nom_f, versets in (("mots_bible_rn.tsv", rn), ("mots_bible_fr.tsv", fr)):
        compte, exemple = compter_mots(versets, ref)
        with open(os.path.join(args.sortie, nom_f), "w", encoding="utf-8") as f:
            f.write("mot\tnombre\tpremière référence\n")
            for mot, n in compte.most_common():
                f.write(f"{mot}\t{n}\t{exemple[mot]}\n")
        print(f"{nom_f} : {len(compte)} mots différents")

    # --- Rapport -------------------------------------------------------------
    lignes = [f"Versets kirundi : {len(rn)}", f"Versets français : {len(fr)}",
              f"Paires alignées gardées : {len(paires)}", "",
              "n°  livre kirundi -> livre français : versets rn / avec fr / gardés  [chapitres rn/fr]"]
    chap_max = collections.defaultdict(lambda: [0, 0])
    for (l, c, v) in rn:
        chap_max[l][0] = max(chap_max[l][0], c)
    for (l, c, v) in fr:
        chap_max[l][1] = max(chap_max[l][1], c)
    for i in range(1, max(len(titres), len(noms)) + 1):
        a = titres[i - 1] if i <= len(titres) else "?"
        b = noms[i - 1] if i <= len(noms) else "?"
        n_rn, n_fr, n_ok = par_livre[i]
        c_rn, c_fr = chap_max[i]
        alerte = ""
        if c_rn != c_fr:
            alerte = "   <-- CHAPITRES DIFFÉRENTS"
        elif n_rn and n_ok / n_rn < 0.8:
            alerte = "   <-- à vérifier"
        lignes.append(f"{i:2d}  {a[:45]:45s} -> {b:18s} : {n_rn:5d} / {n_fr:5d} / {n_ok:5d}  [{c_rn}/{c_fr}]{alerte}")
    lignes += ["", "Exemples :"]
    for pr in paires[:3] + paires[len(paires) // 2: len(paires) // 2 + 3]:
        lignes += [f"[{pr['ref']}]", f"  rn : {pr['rn']}", f"  fr : {pr['fr']}"]
    with open(os.path.join(args.sortie, "rapport.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lignes) + "\n")
    print("\n".join(lignes))


if __name__ == "__main__":
    main()
