"""Télécharge le français : textes pour lire + conversations pour discuter.

    python collecte/telecharger_fr.py --sortie /content/drive/MyDrive/donnees_zero --mots 150000000

1) Wikipédia FR (wikimedia/wikipedia, 20231101.fr) -> lecture_fr/wikipedia_fr_XXX.txt
   On s'arrête à --mots mots (150 millions ~ 200 millions de morceaux, ~12 h sur un T4).
2) OpenAssistant oasst1 : vraies conversations écrites par des humains, en français
   -> dialogues/oasst_fr.jsonl au format {"messages": [...]}.

À lancer sur Colab (il faut la bibliothèque "datasets").
"""

import argparse
import json
import os

from datasets import load_dataset


def wikipedia(sortie, mots_max, par_fichier=20_000_000):
    dossier = os.path.join(sortie, "lecture_fr")
    os.makedirs(dossier, exist_ok=True)
    ds = load_dataset("wikimedia/wikipedia", "20231101.fr", split="train", streaming=True)
    total, n_fichier, dans_fichier, f = 0, 0, 0, None
    for article in ds:
        texte = article["text"].strip()
        if len(texte) < 500:  # pages trop courtes : listes, homonymies
            continue
        if f is None or dans_fichier >= par_fichier:
            if f:
                f.close()
            n_fichier += 1
            f = open(os.path.join(dossier, f"wikipedia_fr_{n_fichier:03d}.txt"), "w", encoding="utf-8")
            dans_fichier = 0
        # Un article = un paragraphe pour train.py (séparé par une ligne vide)
        texte = "\n".join(l for l in texte.splitlines() if l.strip())
        f.write(f"{article['title']}\n{texte}\n\n")
        mots = len(texte.split())
        total += mots
        dans_fichier += mots
        if total >= mots_max:
            break
    f.close()
    print(f"Wikipédia FR : {total:,} mots dans {n_fichier} fichiers")


def oasst(sortie):
    """Reconstruit chaque conversation : question -> meilleure réponse -> question suivante..."""
    dossier = os.path.join(sortie, "dialogues")
    os.makedirs(dossier, exist_ok=True)
    msgs = [m for split in ("train", "validation")
            for m in load_dataset("OpenAssistant/oasst1", split=split) if m["lang"] == "fr"]
    enfants = {}
    for m in msgs:
        enfants.setdefault(m["parent_id"], []).append(m)

    def meilleur(liste):
        liste = [m for m in liste if not m.get("deleted")]
        return min(liste, key=lambda m: m["rank"] if m["rank"] is not None else 99) if liste else None

    n = 0
    with open(os.path.join(dossier, "oasst_fr.jsonl"), "w", encoding="utf-8") as f:
        for racine in enfants.get(None, []):
            conv, m = [], racine
            while m is not None:
                conv.append({"role": "user" if m["role"] == "prompter" else "assistant", "content": m["text"]})
                m = meilleur(enfants.get(m["message_id"], []))
            if conv and conv[-1]["role"] == "user":
                conv.pop()
            if len(conv) >= 2:
                f.write(json.dumps({"messages": conv, "source": "oasst1"}, ensure_ascii=False) + "\n")
                n += 1
    print(f"OpenAssistant FR : {n} conversations")


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--sortie", required=True)
    a.add_argument("--mots", type=int, default=150_000_000)
    args = a.parse_args()
    oasst(args.sortie)
    wikipedia(args.sortie, args.mots)
