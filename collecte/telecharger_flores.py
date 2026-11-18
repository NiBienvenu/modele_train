"""Télécharge FLORES+ (phrases traduites par des professionnels) en kirundi et en français.

FLORES sert seulement de TEST (jamais à l'entraînement) : c'est la vraie preuve de traduction,
sur des phrases qui ne viennent pas de la Bible.

    1. Sur https://huggingface.co/datasets/openlanguagedata/flores_plus : « Agree and access » (une fois).
    2. HF_TOKEN=<jeton de lecture> python3 collecte/telecharger_flores.py
Le jeton est lu dans la variable HF_TOKEN : il n'est écrit dans aucun fichier.
Sortie : data/flores_fr_rn.jsonl  ({"id", "fr", "rn", "partie"})
"""

import json
import os
import sys
import urllib.request
from pathlib import Path

ICI = Path(__file__).resolve().parent.parent
BASE = "https://huggingface.co/datasets/openlanguagedata/flores_plus/resolve/main"


def lire(partie, langue, jeton):
    req = urllib.request.Request(f"{BASE}/{partie}/{langue}.jsonl", headers={"Authorization": f"Bearer {jeton}"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return [json.loads(l) for l in r.read().decode("utf-8").splitlines() if l.strip()]
    except urllib.error.HTTPError as e:
        raise SystemExit(f"Accès refusé ({e.code}) : accepte les conditions sur la page FLORES+ avec ton compte.")


def main():
    jeton = os.environ.get("HF_TOKEN")
    if not jeton:
        raise SystemExit("Mets ton jeton dans HF_TOKEN (voir en haut du fichier).")
    sys.path.insert(0, str(ICI / "collecte"))
    from filtre_kinyarwanda import est_kinyarwanda
    sortie = ICI / "data/flores_fr_rn.jsonl"
    n = rw = 0
    with open(sortie, "w", encoding="utf-8") as out:
        for partie in ("dev", "devtest"):
            fr = {d["id"]: d["text"] for d in lire(partie, "fra_Latn", jeton)}
            for d in lire(partie, "run_Latn", jeton):
                d["text"] = d["text"].replace("\u00b4", "'")  # FLORES écrit kw´utwo : apostrophe normale
                if d["id"] in fr:
                    rw += est_kinyarwanda(d["text"])
                    out.write(json.dumps({"id": f"{partie}-{d['id']}", "fr": fr[d["id"]], "rn": d["text"],
                                          "partie": partie}, ensure_ascii=False) + "\n")
                    n += 1
    print(f"{n} paires dans {sortie} ({rw} phrases kirundi signalées par le filtre kinyarwanda, gardées : "
          "c'est un test de référence)")


if __name__ == "__main__":
    main()
