"""Compare les mots d'ikirundi.org avec les documents (dictionnaire Cox, cours, Bible) et la presse.

    python3 collecte/comparer_sources.py

Pour chaque mot du site :
  connu      : déjà dans les documents (Cox, cours ou Bible) -> le site confirme
  confirmé   : absent des documents, mais aussi dans un texte kirundi indépendant (presse, Wikipédia rn)
               -> accepté comme mot nouveau
  à vérifier : vu SEULEMENT sur le site -> à faire vérifier par un locuteur (data/a_verifier.tsv)

ikirundi.org marque les tons (ô, ā...). Les autres sources non : on les retire pour comparer.
Seuls les mots vus au moins 2 fois sur le site sont comparés (évite les fautes de frappe).
"""

import collections
import json
import re
import unicodedata
from pathlib import Path

from nettoyer_rn import nettoyer_texte

ICI = Path(__file__).resolve().parent.parent
MOT = re.compile(r"[a-zA-Zàâäéèêëîïôöûüçāēīōūǎěǐǒǔ']+")


def sans_tons(t: str) -> str:
    t = unicodedata.normalize("NFD", t.lower().replace("’", "'"))
    return unicodedata.normalize("NFC", "".join(c for c in t if unicodedata.category(c) != "Mn"))


def mots(texte: str):
    for m in MOT.findall(sans_tons(texte)):
        for morceau in m.split("'"):  # n'amahoro -> n, amahoro
            if len(morceau) > 2:
                yield morceau


def tsv_colonnes(chemin: Path, colonnes):
    out = set()
    for l in chemin.open(encoding="utf-8"):
        if l.startswith("#"):
            continue
        champs = l.rstrip("\n").split("\t")
        for c in colonnes:
            if c < len(champs):
                out.update(mots(champs[c]))
    return out


def main():
    documents = set()
    documents |= tsv_colonnes(ICI / "data/noms_cox.tsv", [0, 2])
    documents |= tsv_colonnes(ICI / "data/verbes_cox.tsv", [0, 2])
    documents |= tsv_colonnes(ICI / "data/cours_vocab.tsv", [0])
    documents |= tsv_colonnes(ICI / "A_METTRE_DANS_DRIVE/donnees/bible/mots_bible_rn.tsv", [0])
    for l in (ICI / "data/cours_paires.jsonl").open(encoding="utf-8"):
        documents.update(mots(json.loads(l)["rn"]))

    presse = set()
    for l in (ICI / "data/corpus_rn_propre.jsonl").open(encoding="utf-8"):
        d = json.loads(l)
        if "ikirundi" not in d["source"]:
            presse.update(mots(d["texte"]))

    site, pages = collections.Counter(), collections.defaultdict(set)
    for l in (ICI / "data/ikirundi_org.jsonl").open(encoding="utf-8"):
        d = json.loads(l)
        for m in mots(nettoyer_texte(d["texte"])):
            site[m] += 1
            pages[m].add(d["url"])

    stats = collections.Counter()
    with open(ICI / "data/comparaison_ikirundi.tsv", "w", encoding="utf-8") as f, \
            open(ICI / "data/a_verifier.tsv", "w", encoding="utf-8") as v:
        f.write("# mot\tfois sur le site\tstatut\n")
        v.write("# Mots vus seulement sur ikirundi.org : à vérifier par un locuteur\n# mot\tfois\tune page où il apparaît\n")
        for mot, n in site.most_common():
            if n < 2:
                continue
            statut = "connu" if mot in documents else "confirmé" if mot in presse else "à vérifier"
            stats[statut] += 1
            f.write(f"{mot}\t{n}\t{statut}\n")
            if statut == "à vérifier":
                v.write(f"{mot}\t{n}\t{sorted(pages[mot])[0]}\n")
    print("Mots du site (vus 2 fois ou plus) :", dict(stats))
    print("Détail : data/comparaison_ikirundi.tsv | À vérifier : data/a_verifier.tsv")


if __name__ == "__main__":
    main()
