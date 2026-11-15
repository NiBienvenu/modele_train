"""Garde seulement les textes en kirundi (retire kinyarwanda, anglais, doublons).

    python3 collecte/nettoyer_rn.py data/corpus_rn data/ikirundi_org.jsonl -o data/corpus_rn_propre.jsonl

Kirundi et kinyarwanda sont très proches. Différences d'écriture fiables :
    kirundi : ivyo, vyose, cane, ico, ukwo     (vy, c)
    kinyarwanda : ibyo, byose, cyane, icyo     (by, cy)
Un texte est gardé si les marques kirundi sont au moins aussi nombreuses.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

RN = re.compile(r"\b\w*(vy|c[aeiou])\w*\b", re.I)
RW = re.compile(r"\b\w*(cy|by[aeiou]|jy|shy)\w*\b", re.I)
EN = re.compile(r"\b(the|and|of|is|with|that)\b", re.I)
FR = re.compile(r"\b(le|la|les|des|est|et|une|dans)\b", re.I)
# Texte du lecteur audio d'ikirundi.org (en anglais), à enlever des pages
LECTEUR = {"previous", "play", "next", "stop", "mute", "max volume", "repeat", "shuffle", "update required",
           "to play the media you will need to either update your browser to a recent version or update "
           "your flash plugin."}


def _ligne_lecteur(ligne: str) -> bool:
    l = re.sub(r"\s+", " ", ligne).strip().lower().replace(" .", ".")
    return l in LECTEUR or "flash plugin" in l


def nettoyer_texte(texte: str) -> str:
    return "\n".join(l for l in texte.splitlines() if not _ligne_lecteur(l))


def langue(texte: str) -> str:
    mots = max(1, len(texte.split()))
    if len(EN.findall(texte)) / mots > 0.05:
        return "en"
    if len(FR.findall(texte)) / mots > 0.08:
        return "fr"
    rn, rw = len(RN.findall(texte)), len(RW.findall(texte))
    return "rn" if rn >= rw else "rw"  # égalité (texte court) : la source dit kirundi


def documents(chemins):
    for chemin in chemins:
        p = Path(chemin)
        for f in sorted(p.glob("*.jsonl")) if p.is_dir() else [p]:
            for ligne in f.open(encoding="utf-8"):
                d = json.loads(ligne)
                yield d.get("source") or f.stem, nettoyer_texte(d["texte"])


def main():
    a = argparse.ArgumentParser()
    a.add_argument("entrees", nargs="+")
    a.add_argument("-o", "--sortie", required=True)
    args = a.parse_args()
    vus, stats = set(), {}
    with open(args.sortie, "w", encoding="utf-8") as out:
        for source, texte in documents(args.entrees):
            cle = hashlib.md5(re.sub(r"\W+", "", texte.lower()).encode()).hexdigest()
            l = "doublon" if cle in vus else langue(texte)
            vus.add(cle)
            stats.setdefault(source, {}).setdefault(l, 0)
            stats[source][l] += 1
            if l == "rn":
                out.write(json.dumps({"source": source, "texte": texte}, ensure_ascii=False) + "\n")
    for s, d in stats.items():
        print(s, d)


if __name__ == "__main__":
    main()
