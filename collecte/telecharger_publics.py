"""Télécharge les données publiques trop grosses pour le dossier dataset.

    python telecharger_publics.py --sortie <TRAVAIL>/publics --mots 150000000

Produit :
  lecture_fr/wikipedia_fr_XXX.txt  Wikipédia FR (wikimedia/wikipedia 20231101.fr), --mots mots
  dialogues/oasst_fr.jsonl         OpenAssistant oasst1 : conversations écrites par des humains, en français
  dialogues/piaf.jsonl             PIAF (Etalab) : questions sur un texte, réponse COPIÉE du texte
                                   + exemples où le texte ne contient pas la réponse
  dialogues/aya.jsonl              Aya (Cohere) : instructions écrites par des humains, français et kirundi

Chaque partie déjà faite est sautée (reprise après coupure).
Il faut la bibliothèque "datasets" (déjà là sur Colab et Kaggle).
"""

import argparse
import json
import os
import random

from datasets import load_dataset


def ecrire_jsonl(chemin, convs):
    n = 0
    with open(chemin, "w", encoding="utf-8") as f:
        for c in convs:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
            n += 1
    print(f"{os.path.basename(chemin)} : {n} conversations")


def conv(question, reponse, source):
    return {"messages": [{"role": "user", "content": question}, {"role": "assistant", "content": reponse}],
            "source": source}


def wikipedia(sortie, mots_max, par_fichier=20_000_000):
    dossier = os.path.join(sortie, "lecture_fr")
    if os.path.exists(os.path.join(dossier, "FINI")):
        return
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
        texte = "\n".join(l for l in texte.splitlines() if l.strip())
        f.write(f"{article['title']}\n{texte}\n\n")  # un article = un paragraphe pour train.py
        mots = len(texte.split())
        total += mots
        dans_fichier += mots
        if total >= mots_max:
            break
    f.close()
    open(os.path.join(dossier, "FINI"), "w").close()
    print(f"Wikipédia FR : {total:,} mots dans {n_fichier} fichiers")


def oasst(dossier):
    """Reconstruit chaque conversation : question -> meilleure réponse -> question suivante..."""
    msgs = [m for split in ("train", "validation")
            for m in load_dataset("OpenAssistant/oasst1", split=split) if m["lang"] == "fr"]
    enfants = {}
    for m in msgs:
        enfants.setdefault(m["parent_id"], []).append(m)

    def meilleur(liste):
        liste = [m for m in liste if not m.get("deleted")]
        return min(liste, key=lambda m: m["rank"] if m["rank"] is not None else 99) if liste else None

    for racine in enfants.get(None, []):
        tours, m = [], racine
        while m is not None:
            tours.append({"role": "user" if m["role"] == "prompter" else "assistant", "content": m["text"]})
            m = meilleur(enfants.get(m["message_id"], []))
        if tours and tours[-1]["role"] == "user":
            tours.pop()
        if len(tours) >= 2:
            yield {"messages": tours, "source": "oasst1"}


PAS_DANS_LE_TEXTE = "Le texte ne contient pas cette information."


def piaf():
    """Le comportement voulu pour les documents (lois, décrets) : répondre avec les mots du texte,
    et dire quand le texte ne contient pas la réponse."""
    ds = list(load_dataset("AgentPublic/piaf", split="train"))
    contextes = list({d["context"] for d in ds})
    for d in ds:
        if not d["answers"]["text"]:
            continue
        q = f"Lis ce texte et réponds à la question.\n\nTexte : {d['context']}\n\nQuestion : {d['question']}"
        yield conv(q, d["answers"]["text"][0], "PIAF")
        if random.random() < 0.15:  # même question, mauvais texte -> il doit dire qu'il ne sait pas
            autre = random.choice(contextes)
            if autre != d["context"]:
                q = f"Lis ce texte et réponds à la question.\n\nTexte : {autre}\n\nQuestion : {d['question']}"
                yield conv(q, PAS_DANS_LE_TEXTE, "PIAF (sans réponse)")


def aya():
    """Seulement les lignes écrites en français (fra) ou en kirundi (run)."""
    for split in ("train", "test"):
        for d in load_dataset("CohereLabs/aya_dataset", split=split):
            if d.get("language_code") in ("fra", "run") and d["inputs"].strip() and d["targets"].strip():
                yield conv(d["inputs"].strip(), d["targets"].strip(), f"aya/{d['language_code']}")


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--sortie", required=True)
    a.add_argument("--mots", type=int, default=150_000_000)
    args = a.parse_args()
    random.seed(42)
    dial = os.path.join(args.sortie, "dialogues")
    os.makedirs(dial, exist_ok=True)
    for nom, fn in [("oasst_fr", lambda: oasst(dial)), ("piaf", piaf), ("aya", aya)]:
        chemin = os.path.join(dial, f"{nom}.jsonl")
        if os.path.exists(chemin):
            continue
        try:
            ecrire_jsonl(chemin, fn())
        except Exception as e:  # une source absente ne bloque pas les autres
            print(f"ATTENTION : {nom} non téléchargé ({e})")
            if os.path.exists(chemin):
                os.remove(chemin)
    wikipedia(args.sortie, args.mots)
    # Le lecteur "streaming" de datasets plante parfois en quittant Python : on sort proprement
    os._exit(0)
