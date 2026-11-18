#!/usr/bin/env python
"""
preparer_donnees_zero.py : construit LE dossier unique datasets/dataset.

    python3 collecte/aspirer_ikirundi.py data/ikirundi_org.jsonl     (une fois, ~1 h)
    python3 collecte/telecharger_corpus_rn.py data/corpus_rn          (une fois)
    python3 preparer_donnees_zero.py                                  (tout le reste)

Le même dossier sert partout : Google Drive (Colab), dataset Kaggle, ou sur la machine.
Il contient aussi le code d'entraînement (dataset/code) : pas besoin de GitHub.

Résultat dans datasets/dataset/ :
    lecture_rn/        vrai texte kirundi (presse, Wikipédia rn, web, ikirundi.org, Bible)
    lecture_fr/        Bible FR (Wikipédia FR est ajoutée sur Colab : collecte/telecharger_fr.py)
    lecture_bilingue/  Bible alignée FR/RN + phrases des cours de Cox
    grammaire/         un PETIT échantillon des tableaux (5 %), pour garder les règles
    dialogues/         conversations, traductions, vocabulaire (gen_dialogues.py)
    code/              train.py, train_tokenizer.py, evaluer.py, configs, tests, telecharger_publics.py
Format : texte, un document = un paragraphe (séparé par une ligne vide), comme train.py l'attend.
"""

import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

ICI = Path(__file__).parent
ANCIEN = ICI / "A_METTRE_DANS_DRIVE/donnees"
SORTIE = ICI / "datasets/dataset"


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


LISEZMOI = """# dataset : tout ce qu'il faut pour entraîner le modèle ZÉRO (français + kirundi)

Le même dossier marche partout :
- **Colab** : glisse ce dossier dans Mon Drive, ouvre `notebooks/zero_colab.ipynb`.
- **Kaggle** : crée un dataset privé avec ce dossier, ouvre `notebooks/zero_kaggle.ipynb`.
- **Machine** : ouvre `modele_train/zero_machine.ipynb`.

| Dossier | Contenu |
|---|---|
| lecture_rn/ | vrai texte kirundi : presse (KIRNEWS), Wikipédia rn, web (MADLAD), ikirundi.org, Bible |
| lecture_fr/ | Bible en français (Wikipédia FR est téléchargée par le notebook) |
| lecture_bilingue/ | phrases des cours de Cox et lexique, français + kirundi |
| grammaire/ | un petit échantillon des tableaux de conjugaison et d'accords |
| dialogues/ | salutations, petites conversations, traductions (Bible, cours), vocabulaire, titres d'articles |
| code/ | train.py, configs, tests, téléchargement des données publiques |
| notebooks/ | les 3 notebooks |

Aucune phrase kirundi n'est inventée : elles viennent toutes de textes écrits par des humains
(cours et dictionnaire de Cox, Bible, presse, sites). Ne pas publier ce dossier : les textes
appartiennent à leurs auteurs.

Refaire ce dossier : `python3 preparer_donnees_zero.py` dans modele_train/.
"""


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
    # Pas de Bible bilingue ici : ses versets FR + RN côte à côte contiendraient les versets de TEST.
    # La traduction biblique se fait dans dialogues/traductions_bible.jsonl (versets de test retirés).
    total["bilingue"] = ecrire(SORTIE / "lecture_bilingue/cours_cox.txt",
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

    # 5. Dialogues (et tests)
    subprocess.run([sys.executable, str(ICI / "gen_dialogues.py"), "--sortie", str(SORTIE / "dialogues")], check=True)

    # 6. Le code, pour entraîner sans GitHub
    code = SORTIE / "code"
    for f in ["train.py", "train_tokenizer.py", "evaluer.py", "requirements.txt",
              "configs/zero_pretrain.yaml", "configs/zero_chat.yaml",
              "collecte/telecharger_publics.py", "tests/dialogues_test.jsonl"]:
        (code / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ICI / f, code / f)
    print("Code copié dans", code)
    subprocess.run([sys.executable, str(ICI / "faire_notebooks.py")], check=True)
    (SORTIE / "LISEZMOI.md").write_text(LISEZMOI, encoding="utf-8")

    print("\nTOTAL (mots) :", {k: f"{v:,}" for k, v in total.items()})
    print("Note : fsi_rn.txt est écarté (texte scanné avec beaucoup d'erreurs).")


if __name__ == "__main__":
    main()
