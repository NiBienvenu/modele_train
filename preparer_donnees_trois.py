"""Construit datasets/dataset_trois/ (modèle TROIS).

    python3 preparer_donnees_trois.py

Le trois garde tout ce qu'il y a dans dataset, dataset_un et dataset_deux (Kaggle les lit toujours).
dataset_trois contient seulement le NOUVEAU :
  data/scans_transcrits/Dusome4   livre de lecture 4e année (recopié page par page depuis Dusome_4.pdf)
Une ligne avec [?] (mot douteux) ne va JAMAIS dans l'entraînement : elle va dans a_verifier.tsv.

Il régénère aussi dataset_deux/traduction/bible_choisie.jsonl (Bible corrigée) et met le code
à jour (evaluer.py sans pénalité, configs trois) dans dataset_trois/code/.
"""

import re
import shutil
import subprocess
import sys

import preparer_donnees_deux as deux
from preparer_donnees_deux import ICI, ecrire_jsonl, ecrire_txt, pages, paragraphes

OUT = ICI / "datasets/dataset_trois"


def livre_de_lecture(nom: str, cle: str):
    """Texte du livre + questions sur les mots expliqués (« Insiguro y'amajambo »)."""
    texte, insiguro, dans = [], [], False
    for page, lignes in pages(nom):
        for l in lignes:
            if l.lower().startswith("insiguro y'amajambo"):
                dans = True
                continue
            m = re.match(r"^([^:]{2,40}?)\s*:\s*(.{3,})$", l)
            if dans and m:
                mot, sens = m.group(1).strip(" -"), m.group(2).strip()
                for q in ["Que veut dire « {m} » en kirundi ?", "« {m} » bisobanura iki?"]:
                    insiguro.append({"user": q.format(m=mot), "assistant": f"« {mot} » : {sens}",
                                     "source": f"{cle} {page}"})
                continue
            dans = False
            if not re.fullmatch(r"\d+", l) and l != "[page sans texte]":
                texte.append(l)
    return texte, insiguro


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    # 1. Dusome 4
    d4, ins4 = livre_de_lecture("Dusome4", "Dusome4")
    ecrire_txt(OUT / "lecture_rn/dusome4.txt", d4)
    ecrire_jsonl(OUT / "insiguro/dusome4_insiguro.jsonl", ins4)
    ecrire_txt(OUT / "insiguro/dusome4_insiguro.txt", sorted({o["assistant"] for o in ins4}))
    lecture = paragraphes(d4, "dusome4")
    ecrire_jsonl(OUT / "lecture.jsonl", lecture)

    # 2. Bible corrigée (dans dataset_deux, déjà lu par Kaggle)
    bible = deux.bible_choisie()
    ecrire_jsonl(ICI / "datasets/dataset_deux/traduction/bible_choisie.jsonl", bible)

    # 3. lignes douteuses à relire à la main
    lignes = sorted(a for a in deux.a_verifier if a[0] == "Dusome4")
    (OUT / "a_verifier.tsv").write_text("livre\tpage\tligne\n" + "".join(f"{a}\t{b}\t{c}\n" for a, b, c in lignes),
                                        encoding="utf-8")

    mots = sum(len(l.split()) for l in d4)
    print(f"Dusome 4 : {len(d4)} lignes, {mots} mots, {len(ins4)} questions insiguro, {len(lignes)} lignes à vérifier")
    print(f"Bible : {len(bible)} passages")


def code_et_notebooks():
    code = OUT / "code"
    for f in ["train.py", "evaluer.py", "comparer_resultats.py", "requirements.txt", "configs/trois_lecture.yaml",
              "configs/trois_chat.yaml", "collecte/telecharger_publics.py", "tests/dialogues_test.jsonl"]:
        (code / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ICI / f, code / f)
    shutil.copy2(ICI / "datasets/dataset_deux/tests/flores.jsonl", code / "tests/flores.jsonl")
    subprocess.run([sys.executable, str(ICI / "faire_notebooks_trois.py")], check=True)


if __name__ == "__main__":
    main()
    code_et_notebooks()
