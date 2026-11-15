"""Télécharge des textes kirundi publics, écrits par des humains.

    python3 collecte/telecharger_corpus_rn.py data/corpus_rn

Sources :
- KIRNEWS : ~4 600 articles de presse burundais (Niyongabo et al., 2020).
- MADLAD-400 (partie "rn_clean") : pages web kirundi filtrées (allenai).
- Wikipédia en kirundi (rn.wikipedia.org), copie texte brut sur HuggingFace.
Les textes en kinyarwanda sont retirés plus tard par collecte/nettoyer_rn.py.
Sortie : un fichier .jsonl par source, une ligne {"source", "texte"} par document.
"""

import gzip
import re
import io
import json
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

UA = {"User-Agent": "Mozilla/5.0 (corpus kirundi, usage personnel)"}


def get(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
        return r.read()


def ecrire(chemin: Path, source: str, textes):
    n = 0
    with chemin.open("w", encoding="utf-8") as f:
        for t in textes:
            t = t.strip()
            if len(t) > 50:
                f.write(json.dumps({"source": source, "texte": t}, ensure_ascii=False) + "\n")
                n += 1
    print(source, n, "documents ->", chemin)


def kirnews():
    z = zipfile.ZipFile(io.BytesIO(get("https://github.com/saradhix/kinnews_kirnews/raw/master/KIRNEWS.zip")))
    for nom in z.namelist():
        if nom.endswith(".csv") and "raw" in nom:
            import csv

            lecteur = csv.DictReader(io.TextIOWrapper(z.open(nom), encoding="utf-8"))
            for ligne in lecteur:
                yield f"{ligne.get('title', '')}\n{ligne.get('content', '')}"


def madlad():
    url = "https://huggingface.co/datasets/allenai/MADLAD-400/resolve/main/data/rn/rn_clean_0000.jsonl.gz"
    for ligne in gzip.decompress(get(url)).decode("utf-8").splitlines():
        yield json.loads(ligne)["text"].replace("\\n", "\n")


def wikipedia():
    """Copie texte brut de rn.wikipedia.org (l'API limite trop les requêtes)."""
    url = "https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-kirundi/resolve/main/data/rnwiki_01_of_01.txt.gz"
    texte = gzip.decompress(get(url)).decode("utf-8")
    yield from re.split(r"\n\s*\n(?=\S)", texte)


if __name__ == "__main__":
    dossier = Path(sys.argv[1])
    dossier.mkdir(parents=True, exist_ok=True)
    for nom, fn in [("kirnews", kirnews), ("madlad_rn", madlad), ("wikipedia_rn", wikipedia)]:
        try:
            ecrire(dossier / f"{nom}.jsonl", nom, fn())
        except Exception as e:
            print("ERREUR", nom, e, file=sys.stderr)
