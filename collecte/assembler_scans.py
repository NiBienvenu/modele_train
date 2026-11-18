"""Assemble les transcriptions des PDF scannés (data/scans_transcrits/<livre>/p-XX.txt).

Les pages ont été recopiées à la main (lecture des images), pas par OCR.
    python3 collecte/assembler_scans.py
Sorties :
  data/corpus_un/docs_scannes.jsonl  texte sûr (une page = un document) -> entraînement
  data/scans_a_verifier.tsv          lignes avec [?] (mot douteux), PAS dans l'entraînement
  data/scans_mots_inconnus.tsv       mots jamais vus ailleurs (Bible, Cox, corpus) : à vérifier
  data/insiguro_amajambo.tsv         « Insiguro y'amajambo agoye » des livres : mot -> sens (kirundi)
Un mot expliqué dans insiguro devient connu. Pas de rapprochement automatique des formes
conjuguées : les infixes (ku-bi-teka = guteka + -bi-) le rendent faux (kubiteka n'est pas gukubita).
Double lecture : un mot inconnu doit aussi être lu par l'OCR sur la même page
(data/scans_ocr_pages.json) ou être validé (data/scans_mots_valides.txt, un mot par ligne).
Sinon la ligne va dans scans_a_verifier.tsv, pas dans l'entraînement.
Une ligne corrigée dans scans_transcrits/ (sans [?]) entre dans l'entraînement au prochain lancement.
"""

import json
import re
from collections import Counter
from pathlib import Path

ICI = Path(__file__).resolve().parent.parent
SCANS = ICI / "data/scans_transcrits"
MOT = re.compile(r"[a-zâêîôûáéíóú]+", re.I)


def vocabulaire() -> set:
    vus = set()
    sources = [ICI / "A_METTRE_DANS_DRIVE/donnees/bible/bible_rn.txt", ICI / "data/cours_vocab.tsv",
               ICI / "data/verbes_cox.tsv", ICI / "data/noms_cox.tsv", ICI / "data/corpus_rn_propre.jsonl",
               ICI / "data/corpus_un/docs_ptp.jsonl"]
    for f in sources:
        if f.exists():
            vus.update(m.lower() for m in MOT.findall(f.read_text(encoding="utf-8")))
    return vus


def insiguro() -> list:
    """Sections « Insiguro y'amajambo agoye » : lignes « mot : sens » sous ce titre."""
    entrees = []
    for f in sorted(SCANS.glob("*/p-*.txt")):
        dedans = False
        for ligne in f.read_text(encoding="utf-8").splitlines():
            if "insiguro y'amajambo" in ligne.lower():
                dedans = True
                continue
            m = re.match(r"^(?:\d+\.\s*)?([^:?]{2,40}?)\s+:\s+(.+)$", ligne.strip())
            if dedans and m and "[?]" not in ligne:
                entrees.append((m.group(1).strip(), m.group(2).strip(), f"{f.parent.name}/{f.name}"))
            else:
                dedans = False
    return entrees


def main():
    vus = vocabulaire()
    glossaire = insiguro()
    with open(ICI / "data/insiguro_amajambo.tsv", "w", encoding="utf-8") as out:
        out.write("ijambo\tinsiguro\tfichier\n")
        for t, d, f in glossaire:
            out.write(f"{t}\t{d}\t{f}\n")
    for t, d, _ in glossaire:  # mots du glossaire et de leurs explications = mots connus
        vus.update(m.lower() for m in MOT.findall(t + " " + d))
    ocr = json.loads((ICI / "data/scans_ocr_pages.json").read_text(encoding="utf-8"))
    valides = ICI / "data/scans_mots_valides.txt"
    valides = {l.strip().lower() for l in valides.open(encoding="utf-8")} if valides.exists() else set()
    vus |= valides
    inconnus, exemple = Counter(), {}
    pages = douteuses = 0
    with open(ICI / "data/corpus_un/docs_scannes.jsonl", "w", encoding="utf-8") as out, \
            open(ICI / "data/scans_a_verifier.tsv", "w", encoding="utf-8") as ver:
        ver.write("fichier\tligne\traison\n")
        for f in sorted(SCANS.glob("*/p-*.txt")):
            nom = f"{f.parent.name}/{f.name}"
            gardees, fr = [], False
            lus_ocr = {m.lower() for m in MOT.findall(ocr.get(nom, ""))}
            for ligne in f.read_text(encoding="utf-8").splitlines():
                ligne = ligne.strip()
                if ligne == "[FR]":
                    fr = True
                    continue
                if fr or not ligne:  # la ligne après [FR] est en français
                    fr = False
                    continue
                if "[?]" in ligne:
                    ver.write(f"{nom}\t{ligne}\tmot illisible\n")
                    douteuses += 1
                    continue
                pas_lus = [m for m in MOT.findall(ligne) if m.lower() not in vus and m.lower() not in lus_ocr]
                if pas_lus:
                    ver.write(f"{nom}\t{ligne}\tnon confirmé : {' '.join(pas_lus)}\n")
                    douteuses += 1
                    continue
                gardees.append(ligne)
                for m in MOT.findall(ligne):
                    if m.lower() not in vus and not m[:1].isupper():
                        inconnus[m.lower()] += 1
                        exemple.setdefault(m.lower(), (nom, ligne))
            if gardees:
                out.write(json.dumps({"source": "docs_scannes", "texte": "\n".join(gardees)},
                                     ensure_ascii=False) + "\n")
                pages += 1
    with open(ICI / "data/scans_mots_inconnus.tsv", "w", encoding="utf-8") as out:
        out.write("mot\tfois\tfichier\texemple\tcorrection (laisser vide si juste)\n")
        for m, n in inconnus.most_common():
            out.write(f"{m}\t{n}\t{exemple[m][0]}\t{exemple[m][1]}\t\n")
    print(f"{pages} pages gardées, {douteuses} lignes à vérifier, {len(glossaire)} mots expliqués (insiguro), "
          f"{len(inconnus)} mots inconnus")


if __name__ == "__main__":
    main()
