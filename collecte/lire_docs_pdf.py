"""Extrait le texte kirundi des PDF de docs/ (lecture) vers data/corpus_un/.

- PDF avec texte (PTP_KNDI_*) : pdftotext.
- PDF scannés : PAS ici (l'OCR faisait trop d'erreurs). Voir collecte/assembler_scans.py.
  lire_scan() ci-dessous sert seulement de 2e lecture de contrôle (OCR, rapidocr_onnxruntime).
  On garde seulement les lignes sûres : score OCR >= 0.85, au moins 2 mots,
  pas d'exercices de syllabes (ba-ya-ga), pas de lignes d'une seule lettre répétée.
  L'écriture cursive et le texte vu à travers la page sont ainsi presque tous retirés.

    pip install rapidocr_onnxruntime      (une fois, dans un venv)
    python3 collecte/lire_docs_pdf.py ../docs
Ensuite nettoyer_rn.py (via preparer_donnees_zero.py) retire le kinyarwanda et les doublons.
"""

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ICI = Path(__file__).resolve().parent.parent
SORTIE = ICI / "data/corpus_un"
TEXTE = ["PTP_KNDI_TO.pdf", "PTP_KNDI_FETE_ADVENT_CAREME_PAQUES_FORMAT A5.pdf"]
SCANS = ["dusome-1.pdf", "dusome-2.pdf", "dusome-3-3.pdf", "dusome-3-4.pdf", "Dusome6.pdf",
         "Inyigusho y indero tuntu no gukundisha igihugu.pdf"]


def ligne_ok(t: str, score: float) -> bool:
    t = t.strip()
    mots = t.split()
    if score < 0.85 or len(mots) < 2:
        return False
    if re.search(r"\w-\w|\w ?- ?\w-", t) and len(mots) < 6:  # syllabes : ba-ya-ga
        return False
    if len(set(re.sub(r"\W", "", t.lower()))) <= 2:  # « a a a a »
        return False
    if re.fullmatch(r"[-\d\s.]+", t):  # numéro de page
        return False
    return True


def lire_texte(pdf: Path):
    brut = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True).stdout
    for page in brut.split("\f"):
        lignes = [re.sub(r"\s{2,}", " ", l).strip() for l in page.splitlines()]
        lignes = [l for l in lignes if l and not re.fullmatch(r"[-\d\s.]+", l)]
        if len(lignes) >= 2:
            yield "\n".join(lignes)


def lire_scan(pdf: Path, ocr):
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(["pdftoppm", "-r", "200", "-png", str(pdf), f"{d}/p"], check=True)
        for img in sorted(Path(d).glob("p-*.png")):
            res, _ = ocr(str(img))
            gardees, vues = [], set()
            for _, t, s in res or []:
                if ligne_ok(t, float(s)) and t not in vues:
                    vues.add(t)
                    gardees.append(t.strip())
            if gardees:
                yield "\n".join(gardees)
            print(f"  {pdf.name} {img.stem}: {len(gardees)} lignes", flush=True)


def main():
    docs = Path(sys.argv[1] if len(sys.argv) > 1 else ICI.parent / "docs")
    with open(SORTIE / "docs_ptp.jsonl", "w", encoding="utf-8") as out:
        for nom in TEXTE:
            n = 0
            for page in lire_texte(docs / nom):
                out.write(json.dumps({"source": "docs_ptp", "texte": page}, ensure_ascii=False) + "\n")
                n += 1
            print(nom, n, "pages")
    # Les PDF scannés ne passent PLUS par l'OCR (trop d'erreurs, refusé) : ils sont recopiés
    # page par page dans data/scans_transcrits/ puis assemblés par collecte/assembler_scans.py.
    # (lire_scan() reste ici seulement pour produire la 2e lecture de contrôle.)

if __name__ == "__main__":
    main()
