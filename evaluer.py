#!/usr/bin/env python
"""
evaluer.py : fait passer le jeu de tests (tests/tests.jsonl) à un modèle
entraîné, et donne une note par catégorie.

Deux notes :
  exact : la réponse est EXACTEMENT la bonne (sans compter majuscules et
          ponctuation). C'est la note qui compte pour la conjugaison,
          les accords et les pronoms.
  chrF  : ressemblance lettre par lettre, de 0 à 100. C'est la mesure
          habituelle pour la TRADUCTION (une traduction juste peut être
          dite avec d'autres mots, l'exact est trop sévère).

Il mesure aussi la VITESSE (tokens par seconde), pour vérifier que le
modèle répond vite.

Utilisation dans Colab :
    !python evaluer.py --modele /content/drive/MyDrive/modele/phaseD/finale
    !python evaluer.py --modele ... --categorie traduction_rn_fr --montrer 10
"""

import argparse
import collections
import json
import re
import time
import unicodedata

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def normaliser(t):
    t = unicodedata.normalize("NFC", t).lower().replace("’", "'")
    t = re.sub(r"[^\w' ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def chrf(hyp, ref, n=6, beta=2):
    """chrF : moyenne des n-grammes de caractères (1 à 6), rappel favorisé."""
    hyp, ref = normaliser(hyp).replace(" ", ""), normaliser(ref).replace(" ", "")
    if not hyp or not ref:
        return 0.0
    precisions, rappels = [], []
    for k in range(1, n + 1):
        h = collections.Counter(hyp[i:i + k] for i in range(len(hyp) - k + 1))
        r = collections.Counter(ref[i:i + k] for i in range(len(ref) - k + 1))
        if not h or not r:
            continue
        commun = sum((h & r).values())
        precisions.append(commun / sum(h.values()))
        rappels.append(commun / sum(r.values()))
    if not precisions:
        return 0.0
    p, r = sum(precisions) / len(precisions), sum(rappels) / len(rappels)
    if p + r == 0:
        return 0.0
    return 100 * (1 + beta ** 2) * p * r / (beta ** 2 * p + r)


@torch.no_grad()
def repondre(model, tok, question, appareil, max_tokens=96):
    prompt = f"<|user|> {question} <|assistant|>"
    ids = tok(prompt, return_tensors="pt").to(appareil)
    out = model.generate(**ids, max_new_tokens=max_tokens, do_sample=False,
                         eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    nouveaux = out[0][ids["input_ids"].shape[1]:]
    return tok.decode(nouveaux, skip_special_tokens=True).strip(), len(nouveaux)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modele", required=True)
    p.add_argument("--tests", default="tests/tests.jsonl")
    p.add_argument("--categorie", help="ne tester qu'une catégorie")
    p.add_argument("--max", type=int, default=0, help="nombre max de tests par catégorie (0 = tous)")
    p.add_argument("--montrer", type=int, default=3, help="exemples affichés par catégorie")
    p.add_argument("--sortie", default="tests/resultats.jsonl")
    args = p.parse_args()

    appareil = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.modele)
    model = AutoModelForCausalLM.from_pretrained(args.modele).to(appareil).eval()
    print(f"Modèle : {sum(x.numel() for x in model.parameters()) / 1e6:.1f}M paramètres, sur {appareil}")

    tests = [json.loads(l) for l in open(args.tests, encoding="utf-8")]
    if args.categorie:
        tests = [t for t in tests if t["categorie"] == args.categorie]
    par_cat = collections.defaultdict(list)
    for t in tests:
        par_cat[t["categorie"]].append(t)

    resultats, total_tokens, total_temps = [], 0, 0.0
    print(f"\n{'catégorie':30s} {'tests':>6s} {'exact':>7s} {'chrF':>6s}  (vu à l'entraînement ?)")
    for cat, liste in par_cat.items():
        if args.max:
            liste = liste[: args.max]
        exacts, chrfs = 0, []
        for t in liste:
            debut = time.time()
            rep, n = repondre(model, tok, t["question"], appareil)
            total_temps += time.time() - debut
            total_tokens += n
            ok = normaliser(rep) == normaliser(t["attendu"])
            score = chrf(rep, t["attendu"])
            exacts += ok
            chrfs.append(score)
            resultats.append({**t, "reponse": rep, "exact": ok, "chrf": round(score, 1)})
        vu = "oui" if liste and liste[0]["vu"] else "NON (vraie preuve)"
        print(f"{cat:30s} {len(liste):6d} {100 * exacts / len(liste):6.1f}% {sum(chrfs) / len(chrfs):6.1f}  ({vu})")

    with open(args.sortie, "w", encoding="utf-8") as f:
        for r in resultats:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nVitesse : {total_tokens / max(total_temps, 1e-6):.0f} tokens/seconde sur {appareil}")
    print(f"Détail de chaque réponse : {args.sortie}\n")

    for cat in par_cat:
        exemples = [r for r in resultats if r["categorie"] == cat][: args.montrer]
        if exemples:
            print(f"--- {cat}")
            for r in exemples:
                print(f"  Q : {r['question'][:110]}")
                print(f"  R : {r['reponse'][:110]}")
                print(f"  attendu : {r['attendu'][:110]}   [{'OK' if r['exact'] else 'chrF ' + str(r['chrf'])}]")


if __name__ == "__main__":
    main()
