#!/usr/bin/env python
"""
fusionner_cox.py : ajoute le dictionnaire de Cox à nos données kirundi.

1) data/verbes_rn.tsv
   - les perfectifs trouvés AUTOMATIQUEMENT (Bible, FSI-auto) ou corrigés
     à la main qui ne sont pas d'accord avec Cox sont remplacés par la
     forme de Cox (un vrai dictionnaire) ;
   - les perfectifs vus directement dans le livre FSI (source "FSI") sont
     gardés, même si Cox donne une autre forme (les deux existent sans doute) ;
   - les verbes de data/verbes_a_completer.tsv que Cox connaît y sont
     ajoutés, avec leur sens français.
2) data/verbes_cox.tsv : tous les AUTRES verbes de Cox (sens français
   vide ; le sens anglais est gardé en dernière colonne, pour plus tard).
3) data/noms_cox.tsv : les noms de Cox dont la classe est reconnue
   (même format que data/noms_rn.tsv).
4) data/verbes_a_completer.tsv : réécrit avec ce qui manque encore.

Utilisation :
    python fusionner_cox.py --cox data/cox
"""

import argparse
import os


def lire_tsv(chemin):
    lignes = []
    if os.path.exists(chemin):
        for l in open(chemin, encoding="utf-8"):
            if l.strip() and not l.startswith("#"):
                lignes.append(l.rstrip("\n").split("\t"))
    return lignes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cox", default="data/cox")
    p.add_argument("--verbes", default="data/verbes_rn.tsv")
    p.add_argument("--a_completer", default="data/verbes_a_completer.tsv")
    p.add_argument("--noms", default="data/noms_rn.tsv")
    args = p.parse_args()

    cox_v = {}
    for c in lire_tsv(os.path.join(args.cox, "cox_verbes.tsv")):
        cox_v.setdefault(c[0], {"racine": c[1], "perfs": [], "en": c[3]})["perfs"].append(c[2])

    # --- 1) Corriger et compléter verbes_rn.tsv --------------------------------
    texte = open(args.verbes, encoding="utf-8").read().splitlines()
    nouveau, corriges, connus = [], [], set()
    for ligne in texte:
        if not ligne.strip() or ligne.startswith("#"):
            nouveau.append(ligne)
            continue
        c = ligne.split("\t")
        inf, perf, src = c[0], c[2], (c[4] if len(c) > 4 else "")
        connus.add(inf)
        if inf in cox_v and perf not in cox_v[inf]["perfs"] and src != "FSI":
            corriges.append((inf, perf, cox_v[inf]["perfs"][0]))
            c[2] = cox_v[inf]["perfs"][0]
            c = c[:4] + ["Cox (corrigé)"]
        nouveau.append("\t".join(c))

    ajoutes, reste = [], []
    for c in lire_tsv(args.a_completer):
        inf, fr = c[0], (c[3] if len(c) > 3 else "")
        if inf in cox_v and inf not in connus:
            ajoutes.append(f"{inf}\t{cox_v[inf]['racine']}\t{cox_v[inf]['perfs'][0]}\t{fr}\tCox")
            connus.add(inf)
        elif inf not in connus:
            reste.append(c)
    if ajoutes:
        nouveau.append("# --- perfectifs du dictionnaire de Cox ---")
        nouveau += ajoutes
    with open(args.verbes, "w", encoding="utf-8") as f:
        f.write("\n".join(nouveau) + "\n")

    with open(args.a_completer, "w", encoding="utf-8") as f:
        f.write("# Verbes sans perfectif confirmé (ni FSI, ni Bible, ni Cox).\n")
        f.write("# infinitif\tracine\tperfectif\tfrançais\tnote\n")
        for c in reste:
            f.write("\t".join(c) + "\n")

    # --- 2) Les autres verbes de Cox ---------------------------------------------
    n_v = 0
    with open("data/verbes_cox.tsv", "w", encoding="utf-8") as f:
        f.write("# Verbes du dictionnaire de Cox (sens français pas encore traduit)\n")
        f.write("# infinitif\tracine\tperfectif\tfrançais\tsource\tsens anglais\n")
        for inf, v in cox_v.items():
            if inf not in connus:
                f.write(f"{inf}\t{v['racine']}\t{v['perfs'][0]}\t\tCox\t{v['en']}\n")
                n_v += 1

    # --- 3) Les noms de Cox ----------------------------------------------------------
    deja_noms = {c[0] for c in lire_tsv(args.noms)} | {c[2] for c in lire_tsv(args.noms) if len(c) > 2}
    n_n = 0
    with open("data/noms_cox.tsv", "w", encoding="utf-8") as f:
        f.write("# Noms du dictionnaire de Cox (sens français pas encore traduit)\n")
        f.write("# singulier\tclasse\tpluriel\tclasse\tfrançais_sg\tfrançais_pl\ttype\tsource\tsens anglais\n")
        for c in lire_tsv(os.path.join(args.cox, "cox_noms.tsv")):
            sg, c1, pl, c2, en = c[0], c[1], c[2], c[3], c[4]
            if c1 == "?" or int(c1) > 14 or sg in deja_noms:
                continue
            type_ = "personne" if c1 == "1" else "chose"
            pl_ok = pl if pl and pl != "-" and c2 != "?" else "-"
            f.write(f"{sg}\t{c1}\t{pl_ok}\t{c2 if pl_ok != '-' else '-'}\t\t\t{type_}\tCox\t{en}\n")
            n_n += 1

    print(f"Perfectifs corrigés d'après Cox : {len(corriges)}")
    for inf, a, b in corriges:
        print(f"  {inf:16s} {a:14s} -> {b}")
    print(f"Verbes de ta liste complétés par Cox : {len(ajoutes)}")
    print(f"Verbes encore à compléter : {len(reste)}")
    print(f"data/verbes_cox.tsv : {n_v} verbes")
    print(f"data/noms_cox.tsv   : {n_n} noms")


if __name__ == "__main__":
    main()
