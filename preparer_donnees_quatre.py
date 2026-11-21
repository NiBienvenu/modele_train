"""Construit datasets/dataset_quatre/ (modèle QUATRE).

    python3 preparer_donnees_quatre.py

Le quatre garde dataset, dataset_un, dataset_deux et dataset_trois (Kaggle les lit toujours).
Pas de nouveau kirundi (règle : pas de nouvelle source sans preuve qu'elle est bonne).
dataset_quatre contient :
  traduction/bible_quatre.jsonl  24 000 passages Bible (au lieu de 16 000), moitié dans chaque sens.
                                 Il est DANS dataset_quatre : impossible d'utiliser une vieille copie
                                 (run trois : Kaggle avait encore la Bible de 9 000 lignes).
  attendu.json                   nombre de lignes de chaque fichier, vérifié par le notebook au départ.
  code/ et notebooks/            code corrigé (évaluation honnête, fin de phrase gardée...).
"""

import json
import random
import shutil
import subprocess
import sys

import preparer_donnees_deux as deux
from preparer_donnees_deux import ICI, ecrire_jsonl

OUT = ICI / "datasets/dataset_quatre"
DS = ICI / "datasets"

# Fichiers lus par les configs quatre : le notebook vérifie qu'ils ont ce nombre de lignes sur Kaggle.
A_VERIFIER = [
    "dataset/dialogues/salutations_rn.jsonl", "dataset/dialogues/petite_conversation.jsonl",
    "dataset/dialogues/traductions_cours.jsonl", "dataset/dialogues/vocabulaire.jsonl",
    "dataset_deux/traduction/flores_dev.jsonl", "dataset_deux/traduction/fr_rn.jsonl",
    "dataset_deux/constitutions/memoire_2018.jsonl", "dataset_trois/insiguro/dusome4_insiguro.jsonl",
    "dataset_trois/lecture_rn/dusome4.txt", "dataset_quatre/traduction/bible_quatre.jsonl",
]


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    random.seed(4)                                   # même tirage à chaque fois
    bible = deux.bible_choisie(n=24000)
    ecrire_jsonl(OUT / "traduction/bible_quatre.jsonl", bible)
    vers_fr = sum("français" in (o.get("user") or o["messages"][0]["content"]).split(" : ", 1)[0] for o in bible)
    print(f"Bible : {len(bible)} passages ({vers_fr} vers le français, {len(bible) - vers_fr} vers le kirundi)")

    attendu = {f: sum(1 for l in (DS / f).open(encoding="utf-8") if l.strip()) for f in A_VERIFIER}
    (OUT / "attendu.json").write_text(json.dumps(attendu, indent=1, ensure_ascii=False), encoding="utf-8")
    for f, n in attendu.items():
        print(f"  {n:7d}  {f}")


def code_et_notebooks():
    code = OUT / "code"
    for f in ["train.py", "evaluer.py", "comparer_resultats.py", "requirements.txt", "configs/quatre_lecture.yaml",
              "configs/quatre_chat.yaml", "collecte/telecharger_publics.py", "tests/dialogues_test.jsonl"]:
        (code / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ICI / f, code / f)
    shutil.copy2(DS / "dataset_deux/tests/flores.jsonl", code / "tests/flores.jsonl")
    subprocess.run([sys.executable, str(ICI / "faire_notebooks_quatre.py")], check=True)


if __name__ == "__main__":
    main()
    code_et_notebooks()
