#!/usr/bin/env python
"""Compare deux fichiers de résultats d'evaluer.py, catégorie par catégorie (mêmes tests).

    python comparer_resultats.py ancien.jsonl nouveau.jsonl [--noms un deux]
Seules les questions présentes dans les DEUX fichiers comptent (comparaison juste).
"""
import argparse
import collections
import json


def lire(f):
    # La clé contient l'historique : la même question ("N'amaki?") peut venir après des tours différents.
    # Avant, ces tests s'écrasaient (run trois : 5 salutations comparées au lieu de 10).
    return {(r["categorie"], r.get("historique", ""), r["question"]): r
            for r in map(json.loads, open(f, encoding="utf-8")) if r}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ancien")
    p.add_argument("nouveau")
    p.add_argument("--noms", nargs=2, default=["ancien", "nouveau"])
    a = p.parse_args()
    A, B = lire(a.ancien), lire(a.nouveau)
    communs = sorted(set(A) & set(B))
    par_cat = collections.defaultdict(list)
    for k in communs:
        par_cat[k[0]].append(k)
    n1, n2 = a.noms
    print(f"{'catégorie':28s} {'tests':>5s} | {'exact ' + n1:>12s} {'exact ' + n2:>12s} | "
          f"{'chrF ' + n1:>10s} {'chrF ' + n2:>10s} {'écart':>7s}")
    for cat, ks in par_cat.items():
        e1 = 100 * sum(A[k]["exact"] for k in ks) / len(ks)
        e2 = 100 * sum(B[k]["exact"] for k in ks) / len(ks)
        c1 = sum(A[k]["chrf"] for k in ks) / len(ks)
        c2 = sum(B[k]["chrf"] for k in ks) / len(ks)
        signe = "+" if c2 > c1 + 0.5 else "-" if c2 < c1 - 0.5 else "="
        print(f"{cat:28s} {len(ks):5d} | {e1:11.1f}% {e2:11.1f}% | {c1:10.1f} {c2:10.1f} {c2 - c1:+6.1f} {signe}")
    print(f"\n{len(communs)} questions communes ({len(A)} dans {n1}, {len(B)} dans {n2}).")


if __name__ == "__main__":
    main()
