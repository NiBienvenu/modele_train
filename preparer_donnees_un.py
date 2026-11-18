#!/usr/bin/env python
"""
preparer_donnees_un.py : construit datasets/dataset_un (ce que le MODÈLE UN apprend en plus).

    python3 preparer_donnees_un.py

L'ancien dataset (datasets/dataset, modèle zéro) n'est PAS modifié : le notebook du
modèle un lit les deux dossiers séparément, chacun avec son nombre de relectures.

Contenu (seulement des données vérifiées) :
  lecture_rn/docs_scannes.txt   Dusome 1, 2, 3, 6 et Inyigusho : recopiés page par page, chaque mot
                                connu ou lu pareil 2 fois (lignes douteuses EXCLUES, voir
                                data/scans_a_verifier.tsv)
  lecture_rn/textes_messe.txt   PTP_KNDI (PDF numériques, pas scannés), sans les étiquettes Igit./Icit.
  insiguro/insiguro.txt         « Insiguro y'amajambo agoye » (sens des mots difficiles), format du livre
  dialogues/insiguro.jsonl      question en français -> réponse = la ligne du livre (aucun kirundi inventé)
  tests/flores.jsonl            FLORES+ (si data/flores_fr_rn.jsonl existe) : TEST seulement
  code/, notebooks/             nouveau code (2 GPU, pauses) et notebooks un_*
Tout le kirundi repasse par le filtre kinyarwanda (collecte/nettoyer_rn.py).
"""

import json
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

ICI = Path(__file__).resolve().parent
SORTIE = ICI / "datasets/dataset_un"
DATA = ICI / "data"

LISEZMOI = """# dataset_un : ce que le modèle UN apprend en plus du modèle zéro

Sur Kaggle : dans le dataset **modele-train**, à la racine, à côté de `dataset/` (modèle zéro).
Pour Colab : dans Mon Drive, à côté de `dataset/`. Le modèle un relit les deux.
À chaque nouvel apprentissage : New Version, on remplace seulement `dataset_un/`.

- `lecture_rn/` : textes kirundi vérifiés (livres scannés recopiés et contrôlés, textes de messe)
- `insiguro/` : sens des mots difficiles (Dusome)
- `dialogues/` : questions-réponses sur les insiguro
- `tests/` : FLORES (traduction FR <-> RN par des professionnels), jamais vu à l'entraînement
- `code/` : code d'entraînement (2 GPU, sauvegarde toutes les 4 h, pause avant la fin de session)
- `notebooks/` : un_kaggle, un_colab, un_machine

Refaire ce dossier : `python3 preparer_donnees_un.py` dans modele_train/.
"""


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
    print(f"{str(chemin.relative_to(SORTIE)):32s} {n:6d} documents {mots:9,d} mots")
    return mots


def main():
    random.seed(42)
    if SORTIE.exists():
        shutil.rmtree(SORTIE)

    # 1. Kirundi vérifié, repassé au filtre kinyarwanda et sans doublons
    subprocess.run([sys.executable, "-m", "assembler_scans"], cwd=ICI / "collecte", check=True)
    propre = DATA / "corpus_un_propre.jsonl"
    subprocess.run([sys.executable, str(ICI / "collecte/nettoyer_rn.py"), str(DATA / "corpus_un"),
                    "-o", str(propre)], check=True)
    par_source = {}
    for l in propre.open(encoding="utf-8"):
        d = json.loads(l)
        par_source.setdefault(d["source"], []).append(d["texte"])
    noms = {"docs_scannes": "docs_scannes.txt", "docs_ptp": "textes_messe.txt"}
    # Livret de messe : on enlève seulement les étiquettes « Igit. » / « Icit. » en début de ligne
    # (qui parle), sinon le modèle les recopie dans ses réponses. La phrase kirundi reste entière.
    etiquette = re.compile(r"(?m)^(Igit|Icit)\.\s*")
    par_source["docs_ptp"] = [etiquette.sub("", d) for d in par_source.get("docs_ptp", [])]
    for source, docs in par_source.items():
        ecrire(SORTIE / "lecture_rn" / noms.get(source, source + ".txt"), docs)

    # 2. Insiguro : lecture (format du livre) + questions-réponses
    lignes = [l.split("\t") for l in (DATA / "insiguro_amajambo.tsv").read_text(encoding="utf-8").splitlines()[1:]
              if l.strip()]
    ecrire(SORTIE / "insiguro/insiguro.txt",
           [f"Insiguro y'amajambo agoye\n{mot} : {sens}" for mot, sens, *_ in lignes])
    questions = ["Que veut dire « {m} » en kirundi ?", "Explique le mot kirundi « {m} ».",
                 "Quel est le sens de « {m} » ?"]
    (SORTIE / "dialogues").mkdir(parents=True, exist_ok=True)
    with open(SORTIE / "dialogues/insiguro.jsonl", "w", encoding="utf-8") as f:
        for mot, sens, *_ in lignes:
            for q in questions:  # 3 façons de demander, réponse = la ligne exacte du livre
                f.write(json.dumps({"user": q.format(m=mot), "assistant": f"« {mot} » : {sens}"},
                                   ensure_ascii=False) + "\n")
    print(f"dialogues/insiguro.jsonl         {3 * len(lignes):6d} questions ({len(lignes)} mots)")

    # 3. Tests FLORES (jamais à l'entraînement)
    flores = DATA / "flores_fr_rn.jsonl"
    (SORTIE / "tests").mkdir(parents=True, exist_ok=True)
    if flores.exists():
        paires = [json.loads(l) for l in flores.open(encoding="utf-8")]
        paires = [p for p in paires if p["partie"] == "devtest"]
        random.shuffle(paires)
        with open(SORTIE / "tests/flores.jsonl", "w", encoding="utf-8") as f:
            for p in paires[:300]:
                f.write(json.dumps({"categorie": "flores_rn_fr", "question": f"Traduis en français : {p['rn']}",
                                    "attendu": p["fr"], "vu": False, "source": "FLORES+ " + p["id"]},
                                   ensure_ascii=False) + "\n")
            for p in paires[300:600]:
                f.write(json.dumps({"categorie": "flores_fr_rn", "question": f"Traduis en kirundi : {p['fr']}",
                                    "attendu": p["rn"], "vu": False, "source": "FLORES+ " + p["id"]},
                                   ensure_ascii=False) + "\n")
        print("tests/flores.jsonl               600 tests (300 RN->FR, 300 FR->RN)")
    else:
        print("ATTENTION : pas de data/flores_fr_rn.jsonl (lance collecte/telecharger_flores.py) : "
              "pas de test FLORES.")

    # 4. Le code (nouvelle version) et les notebooks
    code = SORTIE / "code"
    for f in ["train.py", "evaluer.py", "requirements.txt", "configs/un_lecture.yaml", "configs/un_chat.yaml",
              "collecte/telecharger_publics.py", "tests/dialogues_test.jsonl"]:
        (code / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ICI / f, code / f)
    subprocess.run([sys.executable, str(ICI / "faire_notebooks_un.py")], check=True)
    (SORTIE / "LISEZMOI.md").write_text(LISEZMOI, encoding="utf-8")
    print("\nFini :", SORTIE)


if __name__ == "__main__":
    main()
