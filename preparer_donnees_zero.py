#!/usr/bin/env python
"""
preparer_donnees_zero.py : assemble les textes de LECTURE du modèle zéro.

    python3 collecte/aspirer_ikirundi.py data/ikirundi_org.jsonl
    python3 collecte/telecharger_corpus_rn.py data/corpus_rn
    python3 preparer_donnees_zero.py
    python3 gen_dialogues.py --sortie A_METTRE_DANS_DRIVE/donnees_zero/dialogues

Résultat dans A_METTRE_DANS_DRIVE/donnees_zero/ :
    lecture_rn/        vrai texte kirundi (presse, Wikipédia rn, web, ikirundi.org, Bible)
    lecture_fr/        Bible FR (Wikipédia FR est ajoutée sur Colab : collecte/telecharger_fr.py)
    lecture_bilingue/  Bible alignée FR/RN + phrases des cours de Cox
    grammaire/         un PETIT échantillon des tableaux (5 %), pour garder les règles
    dialogues/         (gen_dialogues.py)
Format : texte, un document = un paragraphe (séparé par une ligne vide), comme train.py l'attend.
"""

import json
import random
import subprocess
import sys
from pathlib import Path

ICI = Path(__file__).parent
ANCIEN = ICI / "A_METTRE_DANS_DRIVE/donnees"
SORTIE = ICI / "A_METTRE_DANS_DRIVE/donnees_zero"


def paragraphes(fichier: Path):
    bloc = []
    for ligne in fichier.open(encoding="utf-8"):
        if ligne.strip():
            bloc.append(ligne.rstrip("\n"))
        elif bloc:
            yield "\n".join(bloc)
            bloc = []
    if bloc:
        yield "\n".join(bloc)


def ecrire(chemin: Path, docs):
    chemin.parent.mkdir(parents=True, exist_ok=True)
    n = mots = 0
    with chemin.open("w", encoding="utf-8") as f:
        for d in docs:
            d = "\n".join(l.strip() for l in d.splitlines() if l.strip())
            if d:
                f.write(d + "\n\n")
                n += 1
                mots += len(d.split())
    print(f"{str(chemin.relative_to(SORTIE)):45s} {n:7d} documents {mots:10,d} mots")
    return mots


def main():
    random.seed(42)
    total = {}

    # 1. Kirundi : on filtre (kirundi seulement, sans doublons) puis on écrit
    propre = ICI / "data/corpus_rn_propre.jsonl"
    entrees = [str(ICI / "data/corpus_rn")]
    if (ICI / "data/ikirundi_org.jsonl").exists():
        entrees.append(str(ICI / "data/ikirundi_org.jsonl"))
    subprocess.run([sys.executable, str(ICI / "collecte/nettoyer_rn.py"), *entrees, "-o", str(propre)], check=True)
    par_source = {}
    for l in propre.open(encoding="utf-8"):
        d = json.loads(l)
        par_source.setdefault(d["source"], []).append(d["texte"])
    rn = 0
    for source, docs in par_source.items():
        nom = "ikirundi_org" if "ikirundi" in source else source
        rn += ecrire(SORTIE / f"lecture_rn/{nom}.txt", docs)
    rn += ecrire(SORTIE / "lecture_rn/bible_rn.txt", paragraphes(ANCIEN / "bible/bible_rn.txt"))
    total["kirundi"] = rn

    # 2. Français (le gros vient de Wikipédia, sur Colab)
    total["français"] = ecrire(SORTIE / "lecture_fr/bible_fr.txt", paragraphes(ANCIEN / "bible/bible_fr.txt"))

    # 3. Bilingue
    total["bilingue"] = ecrire(SORTIE / "lecture_bilingue/bible_bilingue.txt",
                               paragraphes(ANCIEN / "bible/bible_bilingue_25.txt"))
    total["bilingue"] += ecrire(SORTIE / "lecture_bilingue/cours_cox.txt",
                                paragraphes(ANCIEN / "cours/cours_bilingue.txt"))
    total["bilingue"] += ecrire(SORTIE / "lecture_bilingue/lexique.txt",
                                paragraphes(ANCIEN / "lexique_rn/lexique_rn.txt"))

    # 4. Grammaire : seulement un échantillon (avant : 80 % des données !)
    g = 0
    for nom, part in [("grammaire_rn/syntaxe_rn_001.txt", 1.0), ("grammaire_rn/accords_rn_001.txt", 0.05),
                      ("conjugaison_rn/conjugaison_rn_001.txt", 0.05)]:
        docs = [p for p in paragraphes(ANCIEN / nom) if random.random() < part]
        g += ecrire(SORTIE / "grammaire" / Path(nom).name, docs)
    conj_fr = [p for f in sorted((ANCIEN / "conjugaison_fr").glob("*.txt")) for p in paragraphes(f)
               if random.random() < 0.01]
    g += ecrire(SORTIE / "grammaire/conjugaison_fr.txt", conj_fr)
    total["grammaire"] = g

    print("\nTOTAL (mots) :", {k: f"{v:,}" for k, v in total.items()})
    print("Note : fsi_rn.txt est écarté (texte scanné avec beaucoup d'erreurs).")


if __name__ == "__main__":
    main()
