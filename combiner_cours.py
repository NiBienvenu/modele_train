#!/usr/bin/env python
"""
combiner_cours.py : réunit les résultats de l'étude des cours de kirundi
d'Elizabeth Cox (Livre I et Livre II) en données d'entraînement.

Entrée : un dossier "resultats" avec, pour chaque partie lue :
  paires_*.jsonl  {"rn", "en", "fr", "source", "fiabilite", "note"}
  vocab_*.tsv     kirundi<TAB>anglais<TAB>français<TAB>source
  notes_*.md      règles de grammaire (copiées dans docs/grammaire/)

Sortie :
  data/cours_paires.jsonl          paires "sûr" (kirundi exact du livre + français)
  data/cours_paires_a_verifier.jsonl  paires douteuses : PAS utilisées, à relire
  data/cours_vocab.tsv             kirundi, français, source
  --sortie/cours_bilingue.txt      phase A : "Kirundi : ... / Français : ..."
  --sft                            phase D : traductions dans les deux sens
  data/verbes_cox.tsv, data/noms_cox.tsv : la colonne "français" est remplie
                                   quand le vocabulaire du cours donne le sens

L'anglais n'est JAMAIS mis dans les données d'entraînement.
"""

import argparse
import glob
import json
import os
import random
import re
import shutil

from separation import paire_de_test


def nettoyer(t):
    return re.sub(r"\s+", " ", (t or "").replace("’", "'")).strip()


def cle_mot(k):
    """'gukora (koze)' -> 'gukora' ; enlève traits et tons simples."""
    k = re.sub(r"\(.*?\)", "", k).strip().lower()
    return k.replace("–", "").replace("-", "").split(",")[0].strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--resultats", required=True)
    p.add_argument("--sortie", required=True)
    p.add_argument("--sft")
    p.add_argument("--graine", type=int, default=42)
    args = p.parse_args()
    hasard = random.Random(args.graine)
    os.makedirs(args.sortie, exist_ok=True)

    # --- Paires ----------------------------------------------------------------
    surs, douteux, tests, vus = [], [], [], set()
    for f in sorted(glob.glob(os.path.join(args.resultats, "paires_*.jsonl"))):
        for ligne in open(f, encoding="utf-8"):
            if not ligne.strip():
                continue
            d = json.loads(ligne)
            rn, fr = nettoyer(d.get("rn")), nettoyer(d.get("fr"))
            if not rn or not fr or (rn, fr) in vus:
                continue
            vus.add((rn, fr))
            pr = {"rn": rn, "fr": fr, "source": d.get("source", ""), "note": d.get("note", "")}
            if not d.get("fiabilite", "").startswith("s"):
                douteux.append(pr)
            elif paire_de_test(rn):
                tests.append(pr)          # jamais vue à l'entraînement
            else:
                surs.append(pr)

    with open("data/cours_paires.jsonl", "w", encoding="utf-8") as f:
        for pr in surs:
            f.write(json.dumps(pr, ensure_ascii=False) + "\n")
    os.makedirs("tests", exist_ok=True)
    with open("tests/paires_test.jsonl", "w", encoding="utf-8") as f:
        for pr in tests:
            f.write(json.dumps(pr, ensure_ascii=False) + "\n")
    with open("data/cours_paires_a_verifier.jsonl", "w", encoding="utf-8") as f:
        for pr in douteux:
            f.write(json.dumps(pr, ensure_ascii=False) + "\n")

    melange = surs[:]
    hasard.shuffle(melange)
    blocs = []
    for i in range(0, len(melange), 6):
        lignes = []
        for pr in melange[i:i + 6]:
            if hasard.random() < 0.5:
                lignes.append(f"Kirundi : {pr['rn']}\nFrançais : {pr['fr']}")
            else:
                lignes.append(f"Français : {pr['fr']}\nKirundi : {pr['rn']}")
        blocs.append("\n".join(lignes))
    with open(os.path.join(args.sortie, "cours_bilingue.txt"), "w", encoding="utf-8") as f:
        f.write("\n\n".join(blocs) + "\n")

    if args.sft:
        os.makedirs(os.path.dirname(args.sft) or ".", exist_ok=True)
        with open(args.sft, "w", encoding="utf-8") as f:
            for pr in surs:   # les deux sens pour chaque paire
                f.write(json.dumps({"user": f"Traduis en français : {pr['rn']}", "assistant": pr["fr"]}, ensure_ascii=False) + "\n")
                f.write(json.dumps({"user": f"Traduis en kirundi : {pr['fr']}", "assistant": pr["rn"]}, ensure_ascii=False) + "\n")

    # --- Vocabulaire -------------------------------------------------------------
    vocab = {}
    for f in sorted(glob.glob(os.path.join(args.resultats, "vocab_*.tsv"))):
        for ligne in open(f, encoding="utf-8"):
            c = ligne.rstrip("\n").split("\t")
            if len(c) < 3 or not c[0].strip() or not c[2].strip():
                continue
            rn, fr = nettoyer(c[0]), nettoyer(c[2])
            src = c[3].strip() if len(c) > 3 else ""
            vocab.setdefault(cle_mot(rn), (rn, fr, src))
    with open("data/cours_vocab.tsv", "w", encoding="utf-8") as f:
        f.write("# kirundi\tfrançais\tsource\n")
        for rn, fr, src in vocab.values():
            f.write(f"{rn}\t{fr}\t{src}\n")

    # --- Remplir le français des verbes et noms de Cox ------------------------------
    remplis = {}
    for chemin, col_fr in (("data/verbes_cox.tsv", 3), ("data/noms_cox.tsv", 4)):
        if not os.path.exists(chemin):
            continue
        lignes, n = [], 0
        for ligne in open(chemin, encoding="utf-8"):
            if ligne.startswith("#") or not ligne.strip():
                lignes.append(ligne.rstrip("\n"))
                continue
            c = ligne.rstrip("\n").split("\t")
            k = cle_mot(c[0])
            if k in vocab and len(c) > col_fr and not c[col_fr].strip():
                c[col_fr] = vocab[k][1]
                n += 1
            lignes.append("\t".join(c))
        with open(chemin, "w", encoding="utf-8") as f:
            f.write("\n".join(lignes) + "\n")
        remplis[chemin] = n

    # --- Notes de grammaire -> docs/grammaire --------------------------------------
    os.makedirs("docs/grammaire", exist_ok=True)
    for f in glob.glob(os.path.join(args.resultats, "notes_*.md")):
        shutil.copy(f, os.path.join("docs/grammaire", os.path.basename(f)))

    print(f"Paires sûres pour l'entraînement : {len(surs)} ; gardées pour les TESTS : {len(tests)} ; "
          f"à vérifier (non utilisées) : {len(douteux)}")
    print(f"Vocabulaire avec français : {len(vocab)}")
    for chemin, n in remplis.items():
        print(f"Sens français ajoutés dans {chemin} : {n}")
    print("Notes de grammaire copiées dans docs/grammaire/")


if __name__ == "__main__":
    main()
