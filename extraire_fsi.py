#!/usr/bin/env python
"""
extraire_fsi.py : récupère le kirundi du cours FSI "Kirundi Basic Course".

Source : Foreign Service Institute (gouvernement américain, années 1960).
Le cours est dans le DOMAINE PUBLIC : on peut l'utiliser librement.
Page : https://www.fsi-language-courses.org/fsi-kirundi-basic-course/

Le PDF contient déjà du texte (un ancien scan avec OCR), mais ce texte a
des défauts. Le script les corrige :

  1) Voyelles longues écrites en double : "kugeenda", "yaagiiye".
     L'orthographe moderne du kirundi ne marque pas la longueur :
     on écrit "kugenda", "yagiye". On réduit donc "aa" -> "a", etc.
  2) Erreurs de scan : le kirundi n'a PAS de lettre "l". Un "l" dans un
     mot kirundi est presque toujours un "i" mal lu : "lndlmu" -> "indimu".
     Pareil pour "1" collé dans un mot.
  3) Accents de ton (é, á, â...) : retirés, comme dans l'écriture courante.
  4) Phonétique entre [ ] et / / : retirée.
  5) En-têtes de page ("UNIT 15 KIRUNDI"), numéros de page : retirés.

Ce que le script produit (dans --sortie) :
  - fsi_rn.txt          : phrases en kirundi, pour la phase A
  - vocabulaire_fsi.tsv : kirundi <TAB> anglais (mot et traduction)
  - verbes_fsi.tsv      : infinitif <TAB> forme conjuguée <TAB> exercice
                          -> sert à trouver les règles de conjugaison
  - mots_frequents.tsv  : tous les mots kirundi trouvés, avec leur nombre
  - fsi_explications_en.txt : les explications de grammaire en anglais
                          (pas pour l'entraînement, pour nous aider)
  - rapport.txt         : chiffres et exemples pour vérifier à l'oeil

Utilisation dans Colab :
    !pip install pypdf
    !python extraire_fsi.py --sortie /content/drive/MyDrive/donnees/fsi_rn
"""

import argparse
import collections
import os
import re
import unicodedata
import urllib.request

from pypdf import PdfReader

URL_PDF = ("https://fsi-language-courses-media.nyc3.cdn.digitaloceanspaces.com/"
           "languages/Kirundi/Basic/FSI%20-%20Kirundi%20Basic%20Course%20-%20Student%20Text.pdf")

# Mots anglais très fréquents : servent à reconnaître une ligne en anglais.
MOTS_ANGLAIS = set("""
the a an of to in is are was were be been and or but not that this these
those with for from by on at as it its he she they we you i his her their
our your which who what when where how why there here have has had do does
did will would can could should may might must if then than so such also
only some any all each every other more most very into out up about after
before over under between through form forms verb verbs noun nouns note
notes unit exercise tense prefix stem subject object word words sentence
following same used use meaning example examples page see one two three
to fto ito cause caused want wants go goes come came get got make made
take give say see know think work read write eat drink sleep sit stand
child children man men woman women house book books thing things day
time year years good bad big small new old who's i'm don't it's
""".split())

# Le début du livre a quelques pages en français : ces mots finissent
# aussi par une voyelle, il faut donc les reconnaître à part.
MOTS_FRANCAIS = set("""
le la les une des que qui pour dans avec est sont doit avoir donner
cette ce ces elle il ils nous vous leur leurs sur par pas plus tout
etre faire comme mais ou du au aux sa se ne
""".split())

# Morceaux typiques d'une traduction anglaise mal scannée : 'to ...'  !to  Ito
TRADUCTION = re.compile(r"(^|\s)[!'I1f|]?to\s|\s'\S|\S'\s*$|(^|\s)I(\s|$)")

VOYELLES = "aeiou"


# ---------------------------------------------------------------------------
# 1. Télécharger et lire le PDF
# ---------------------------------------------------------------------------

def telecharger(dossier):
    chemin = os.path.join(dossier, "fsi_kirundi.pdf")
    if not os.path.exists(chemin):
        print("Téléchargement du PDF (environ 18 Mo)...")
        urllib.request.urlretrieve(URL_PDF, chemin)
    return chemin


def lire_pages(chemin):
    """Retourne la liste du texte de chaque page."""
    lecteur = PdfReader(chemin)
    print(f"{len(lecteur.pages)} pages")
    return [(p.extract_text() or "") for p in lecteur.pages]


# ---------------------------------------------------------------------------
# 2. Nettoyer
# ---------------------------------------------------------------------------

def enlever_accents(texte):
    """é -> e, â -> a ... (les tons ne sont pas écrits en kirundi courant)."""
    decompose = unicodedata.normalize("NFD", texte)
    sans = "".join(c for c in decompose if unicodedata.category(c) != "Mn")
    return unicodedata.normalize("NFC", sans)


def nettoyer_ligne(ligne):
    ligne = re.sub(r"\[[^\]]*\]", " ", ligne)         # [phonétique]
    ligne = re.sub(r"^\s*(UNIT\s+\d+|SYNOPSIS)\s+KIRUNDI\s*$", "", ligne, flags=re.I)
    ligne = re.sub(r"^\s*KIRUNDI\s+(UNIT\s+\d+)?\s*$", "", ligne, flags=re.I)
    ligne = re.sub(r"^\s*\d{1,3}\s*$", "", ligne)      # numéro de page seul
    ligne = re.sub(r"[.•]{4,}.*$", "", ligne)          # pointillés de la table des matières
    ligne = ligne.replace("/", " ").replace("~", " ").replace("­", "")
    ligne = re.sub(r"(?<=\s)[,`'\"^_-]+(?=\s)", " ", ligne)  # marques de ton isolées
    ligne = re.sub(r"^[\s,`'\"^_.-]+", "", ligne)
    return re.sub(r"\s+", " ", ligne).strip()


def corriger_mot_kirundi(mot):
    """
    Corrige un mot qu'on sait être du kirundi.
      "lndlmu" -> "indimu"  (pas de "l" en kirundi, c'est un "i" mal lu)
      "kuzlmira" -> "kuzimira"
      "kugeenda" -> "kugenda"  (voyelles longues)
    """
    debut_majuscule = mot[:1].isupper()
    m = enlever_accents(mot).lower()
    m = m.replace("l", "i").replace("1", "i").replace("!", "i").replace("|", "i")
    m = re.sub(r"([aeiou])\1+", r"\1", m)  # aa -> a, iii -> i
    m = re.sub(r"[^a-z']", "", m)
    if debut_majuscule and m:
        m = m[0].upper() + m[1:]
    return m


# ---------------------------------------------------------------------------
# 3. Reconnaître la langue d'une ligne
# ---------------------------------------------------------------------------

def mots_de(ligne):
    return re.findall(r"[A-Za-zÀ-ÿ1!|']+", ligne)


def est_kirundi(ligne):
    """
    Une ligne est du kirundi si :
      - presque tous ses mots finissent par une voyelle (c'est le cas en
        kirundi : "Abana baragiye ku ishure"),
      - et elle a très peu de mots anglais courants.
    """
    if TRADUCTION.search(ligne):
        return False
    mots = [m for m in mots_de(ligne) if len(m) > 1]
    if len(mots) < 1:
        return False
    propres = [enlever_accents(m).lower().rstrip("'") for m in mots]
    fin_voyelle = sum(1 for m in propres if m and m[-1] in VOYELLES) / len(propres)
    anglais = sum(1 for m in propres if m in MOTS_ANGLAIS) / len(propres)
    francais = sum(1 for m in propres if m in MOTS_FRANCAIS) / len(propres)
    # L'index à la fin du livre : beaucoup de "i" isolés (restes de tons).
    i_seuls = sum(1 for m in mots_de(ligne) if m.lower() in ("i", "1", "l")) / len(mots_de(ligne))
    return (fin_voyelle >= 0.8 and anglais <= 0.1 and francais <= 0.1
            and i_seuls <= 0.2)


def phrase_kirundi(ligne):
    """Corrige chaque mot et garde la ponctuation de fin."""
    fin = ligne.rstrip()[-1:] if ligne.rstrip()[-1:] in ".?!" else ""
    mots = [corriger_mot_kirundi(m) for m in mots_de(ligne)]
    # On enlève les lettres seules (restes de marques de ton), sauf
    # "i" ("i Bujumbura") et "n'" / "y'" ... collés au mot suivant.
    mots = [m for m in mots if len(m) > 1 or m.lower() == "i"]
    if not mots:
        return ""
    return " ".join(mots) + fin


# ---------------------------------------------------------------------------
# 4. Trouver le vocabulaire et les verbes
# ---------------------------------------------------------------------------

# Exemple : "umwaka 'year'"   ou, mal scanné :  "umwaka !yeart"  "kugenda Ito got"
VOCAB = re.compile(r"^([A-Za-zÀ-ÿ'1 ]{2,40}?)\s+['!I1l|]\s?([a-z][a-z ,'()-]{1,40}?)\s?['!I1l|t]?\s*$")

# Exemple : "kugeenda Yaagiiye?"  (infinitif puis forme conjuguée)
VERBE = re.compile(r"^(?:[,`'\s]*)((?:ku|gu|kw|gw|kuu|guu)[a-zA-ZÀ-ÿ1]{2,})\s+([A-Z][A-Za-zÀ-ÿ1' ]{2,60}[?.!]?)$")

EXERCICE = re.compile(r"^Exercise\s+\d+\.?\s*(.*)$", re.I)


def extraire(pages):
    phrases, vocab, verbes, anglais = [], [], [], []
    compte = collections.Counter()
    exercice = ""

    for page in pages:
        for brute in page.splitlines():
            m_ex = EXERCICE.match(brute.strip())
            if m_ex:
                exercice = m_ex.group(1).strip()[:80]

            ligne = nettoyer_ligne(brute)
            if len(ligne) < 3:
                continue

            # Paires infinitif / forme conjuguée
            m = VERBE.match(ligne)
            if m and est_kirundi(m.group(2)):
                inf = corriger_mot_kirundi(m.group(1))
                forme = phrase_kirundi(m.group(2))
                if forme:
                    verbes.append((inf, forme, exercice))
                    phrases.append(forme)
                    continue

            # Vocabulaire : kirundi + traduction anglaise
            m = VOCAB.match(ligne)
            if m and est_kirundi(m.group(1)) and not est_kirundi(m.group(2)):
                rn = phrase_kirundi(m.group(1))
                en = m.group(2).strip(" '")
                if rn and "'" not in en and len(rn.split()) <= 3:
                    vocab.append((rn, en))
                    continue

            if est_kirundi(ligne):
                p = phrase_kirundi(ligne)
                # On garde les phrases d'au moins 2 mots (les mots seuls
                # sont déjà dans le vocabulaire et le compteur).
                if p:
                    for mot in p.split():
                        compte[mot.lower().strip("'")] += 1
                    if len(p.split()) >= 2:
                        phrases.append(p)
            else:
                anglais.append(ligne)

    return phrases, vocab, verbes, anglais, compte


def sans_doublons(liste):
    vu, sortie = set(), []
    for x in liste:
        if x and x not in vu:
            vu.add(x)
            sortie.append(x)
    return sortie


# ---------------------------------------------------------------------------
# 5. Programme principal
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sortie", required=True)
    p.add_argument("--pdf", help="chemin du PDF s'il est déjà téléchargé")
    args = p.parse_args()
    os.makedirs(args.sortie, exist_ok=True)

    pdf = args.pdf or telecharger(args.sortie)
    pages = lire_pages(pdf)
    phrases, vocab, verbes, anglais, compte = extraire(pages)

    phrases = sans_doublons(phrases)
    vocab = sans_doublons(vocab)
    verbes = sans_doublons(verbes)

    def ecrire(nom, lignes):
        with open(os.path.join(args.sortie, nom), "w", encoding="utf-8") as f:
            f.write("\n".join(lignes) + "\n")

    # Phase A : groupes de 10 phrases = un paragraphe (une ligne vide entre).
    paragraphes = [" ".join(phrases[i:i + 10]) for i in range(0, len(phrases), 10)]
    with open(os.path.join(args.sortie, "fsi_rn.txt"), "w", encoding="utf-8") as f:
        f.write("\n\n".join(paragraphes) + "\n")

    ecrire("vocabulaire_fsi.tsv", [f"{rn}\t{en}" for rn, en in vocab])
    ecrire("verbes_fsi.tsv", [f"{i}\t{f}\t{e}" for i, f, e in verbes])
    ecrire("mots_frequents.tsv", [f"{m}\t{n}" for m, n in compte.most_common()])
    ecrire("fsi_explications_en.txt", anglais)

    rapport = [
        f"Pages lues            : {len(pages)}",
        f"Phrases kirundi       : {len(phrases)}",
        f"Mots de vocabulaire   : {len(vocab)}",
        f"Paires verbe / forme  : {len(verbes)}",
        f"Mots kirundi différents : {len(compte)}",
        "", "Exemples de phrases :", *phrases[200:215],
        "", "Exemples de vocabulaire :", *[f"{a} = {b}" for a, b in vocab[:15]],
        "", "Exemples de verbes :", *[f"{a} -> {b}   ({c})" for a, b, c in verbes[:20]],
        "", "Mots les plus fréquents :", ", ".join(m for m, _ in compte.most_common(60)),
    ]
    ecrire("rapport.txt", rapport)
    print("\n".join(rapport))
    print(f"\nFichiers écrits dans {args.sortie}")


if __name__ == "__main__":
    main()
