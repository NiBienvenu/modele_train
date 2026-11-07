#!/usr/bin/env python
"""
construire_tests.py : fabrique le jeu de tests (tests/tests.jsonl).

Chaque test : {"categorie", "question", "attendu", "vu"}
  "vu" = False : le modèle n'a JAMAIS vu la réponse à l'entraînement
               (paires de traduction et verbes réservés par separation.py)
  "vu" = True  : la réponse est dans les données (on vérifie qu'il a bien appris)

Catégories :
  traduction_rn_fr, traduction_fr_rn  : phrases des cours de Cox jamais vues
  conjugaison_verbe_nouveau           : verbes jamais vus (preuve qu'il a compris les règles)
  conjugaison_verbe_connu             : verbes vus
  accords                             : classe d'un nom, nom + adjectif
  pronoms_objets                      : "Urabona igitabo?" -> "Ego, ndakibona."
  vocabulaire                         : "Comment dit-on ... en kirundi ?" / "Que veut dire ... ?"

Utilisation :  python construire_tests.py   (après combiner_cours.py)
"""

import json
import os
import random

from gen_accords_rn import CLASSES, adjectif, lire_noms
from gen_conjugaison_rn import conjuguer, lire_verbes
from gen_syntaxe_rn import OBJET_CLASSE, present_obj
from separation import verbe_de_test

TEMPS_TESTES = ["présent", "futur (après aujourd'hui)", "passé", "négatif présent",
                "subjonctif (que je ...)", "conditionnel"]


def main():
    hasard = random.Random(7)
    tests = []

    # 1) Traductions jamais vues
    if os.path.exists("tests/paires_test.jsonl"):
        for l in open("tests/paires_test.jsonl", encoding="utf-8"):
            pr = json.loads(l)
            tests.append({"categorie": "traduction_rn_fr", "question": f"Traduis en français : {pr['rn']}",
                          "attendu": pr["fr"], "vu": False})
            tests.append({"categorie": "traduction_fr_rn", "question": f"Traduis en kirundi : {pr['fr']}",
                          "attendu": pr["rn"], "vu": False})

    # 2) Conjugaison : verbes nouveaux (jamais vus) et verbes connus
    verbes = [v for f in ("data/verbes_rn.tsv", "data/verbes_cox.tsv") if os.path.exists(f)
              for v in lire_verbes(f)]
    nouveaux = [v for v in verbes if verbe_de_test(v["inf"])]
    connus = [v for v in verbes if not verbe_de_test(v["inf"]) and v["fr"]]
    for groupe, liste, n in (("conjugaison_verbe_nouveau", nouveaux, 120), ("conjugaison_verbe_connu", connus, 60)):
        for v in hasard.sample(liste, min(n, len(liste))):
            temps = hasard.choice(TEMPS_TESTES)
            formes = [f for f in conjuguer(v["racine"], v["perf"])[temps] if f]
            tests.append({"categorie": groupe, "question": f"Conjugue le verbe {v['inf']} au {temps}.",
                          "attendu": ", ".join(formes) + ".", "vu": groupe.endswith("connu")})

    # 3) Accords et 4) pronoms objets (noms avec sens français)
    noms = [n for n in lire_noms("data/noms_rn.tsv") if n["fr"]]
    for n in hasard.sample(noms, min(40, len(noms))):
        tests.append({"categorie": "accords", "question": f"De quelle classe est le nom {n['mot']} ?",
                      "attendu": f"{n['mot'][0].upper() + n['mot'][1:]} est de la classe {n['classe']}.", "vu": True})
        a = adjectif(n["classe"], "iza")
        if a:
            tests.append({"categorie": "accords", "question": f"Comment dit-on « {n['fr']} » + « bon, beau » en kirundi ?",
                          "attendu": f"{n['mot']} {a}.", "vu": True})
    bona = {"racine": "bona", "perf": "bonye"}
    for n in hasard.sample(noms, min(30, len(noms))):
        if n["classe"] in OBJET_CLASSE:
            tests.append({"categorie": "pronoms_objets",
                          "question": f"Réponds en kirundi avec un pronom : Urabona {n['mot']}?",
                          "attendu": f"Ego, {present_obj('n', OBJET_CLASSE[n['classe']], bona)}.", "vu": True})

    # 5) Vocabulaire courant (dictionnaire français -> kirundi)
    if os.path.exists("data/dictionnaire_verbes_fr_rn.json"):
        dico = json.load(open("data/dictionnaire_verbes_fr_rn.json", encoding="utf-8"))
        simples = [(e["francais"], e["kirundi"][0]) for e in dico if e["kirundi"] and e["kirundi"][0] and " " not in e["kirundi"][0]]
        for fr, rn in hasard.sample(simples, min(60, len(simples))):
            tests.append({"categorie": "vocabulaire", "question": f"Comment dit-on « {fr} » en kirundi ?",
                          "attendu": f"On dit {rn}.", "vu": True})

    os.makedirs("tests", exist_ok=True)
    with open("tests/tests.jsonl", "w", encoding="utf-8") as f:
        for t in tests:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    from collections import Counter
    c = Counter(t["categorie"] for t in tests)
    print(f"{len(tests)} tests écrits dans tests/tests.jsonl")
    for k, n in c.items():
        print(f"  {k:28s} {n}")


if __name__ == "__main__":
    main()
