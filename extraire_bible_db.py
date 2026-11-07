#!/usr/bin/env python
"""
extraire_bible_db.py : lit la Bible kirundi et la Bible française depuis
leurs bases SQLite (bible_kirundi.db, bible_francais.db), les nettoie et
les aligne.

Les deux bases ont la même structure (table "verses" : book, chapter,
verse, text) et les mêmes codes de livres (Gn, Ex, Jos...). Un verset
est aligné quand (livre, chapitre, verset) existe des deux côtés.

POURQUOI DES PASSAGES ET PAS SEULEMENT DES VERSETS ?
  En kirundi, un pronom dans le verbe reprend un nom déjà cité :
      Jos 18,6  Muzoshushanya ico GIHUGU, mu-GI-cemwo ibice indwi...
      Jos 18,5  Bazo-KI-gaburamwo ibice indwi...
  Le français, lui, répète souvent le nom ("le pays"). Si le modèle voit
  un verset seul, il ne sait pas à quoi -ki- renvoie. Avec un PASSAGE de
  plusieurs versets qui se suivent, il voit le nom ET le pronom : il
  apprend la règle des accords d'objet.

NETTOYAGE
  Kirundi : retours à la ligne -> espace ; numéros de versets restés
            dans le texte ("3 Ubu rero", "4Ubwoba") supprimés ; mots
            collés séparés ("ntitwasavyeUmukama" -> "ntitwasavye Umukama") ;
            tons retirés (sauf --garder_tons).
  Français : espaces insécables, crochets d'éditeur [ ] retirés.

TOUT est lu et sert aux listes de mots. Pour l'entraînement, --part
(25 % par défaut) choisit une partie des passages, répartie dans
TOUS les livres (un passage sur quatre), pour que la Bible ne prenne
pas trop de place dans le mélange (sinon le modèle parle "biblique").

FICHIERS PRODUITS (--sortie)
  bible_rn.txt / bible_fr.txt        : texte COMPLET, un paragraphe par passage
  bible_passages.jsonl               : TOUS les passages alignés {ref, rn, fr}
  bible_bilingue_25.txt              : la part choisie, pour la phase A
  mots_bible_rn.tsv / mots_bible_fr.tsv : tous les mots, nombre, référence
  rapport.txt
  (--sft) exercices de traduction (passages de la part choisie)

Utilisation dans Colab :
    !python extraire_bible_db.py \
        --rn /content/drive/MyDrive/sources/bible_kirundi.db \
        --fr /content/drive/MyDrive/sources/bible_francais.db \
        --sortie /content/drive/MyDrive/donnees/bible \
        --sft /content/drive/MyDrive/donnees/qr/traduction_bible.jsonl
"""

import argparse
import collections
import json
import os
import random
import re
import sqlite3
import unicodedata


# ---------------------------------------------------------------------------
# Lecture
# ---------------------------------------------------------------------------

def cle_verset(v):
    """'12' -> (12, '') ; '1a' -> (1, 'a') : pour trier dans le bon ordre."""
    m = re.match(r"(\d+)(\D*)", v or "0")
    return (int(m.group(1)), m.group(2)) if m else (0, v)


def lire_db(chemin):
    """
    Retourne une liste ordonnée de (livre, chapitre, verset, texte) et le
    titre de chaque livre.
    """
    con = sqlite3.connect(chemin)
    lignes = con.execute(
        "SELECT book, book_id, book_title, chapter, chapter_id, verse, text FROM verses").fetchall()
    con.close()
    titres = {}
    rangs = []
    for book, book_id, titre, chap, chap_id, vers, texte in lignes:
        # "3-04" et "3-4" sont le même verset : on enlève les zéros inutiles
        vers = re.sub(r"(?<!\d)0+(?=\d)", "", str(vers))
        titres.setdefault(book, titre)
        rangs.append(((book_id, int(chap) if str(chap).isdigit() else chap_id, cle_verset(vers)),
                      (book, str(chap), str(vers), texte or "")))
    rangs.sort(key=lambda x: x[0])
    return [r[1] for r in rangs], titres


# ---------------------------------------------------------------------------
# Nettoyage
# ---------------------------------------------------------------------------

def sans_tons(t):
    d = unicodedata.normalize("NFD", t)
    d = "".join(c for c in d if c not in "̆̄́̀̂̌")
    return unicodedata.normalize("NFC", d)


def nettoyer_rn(t, garder_tons=False):
    t = unicodedata.normalize("NFC", t).replace("\r", " ").replace("\n", " ").replace("\t", " ")
    t = t.replace("’", "'").replace("‘", "'")
    # Numéros de versets restés dans le texte : "3 Ubu rero", "4Ubwoba", "... mwenewabo. 1"
    # (le kirundi écrit les nombres en toutes lettres dans la Bible)
    t = re.sub(r"(?<![\w])\d{1,3}(?=[A-Za-zÀ-ÿ])", "", t)
    t = re.sub(r"(?<![\w])\d{1,3}(?![\w])", "", t)
    # Mots collés : "ntitwasavyeUmukama" -> "ntitwasavye Umukama"
    t = re.sub(r"(?<=[a-zà-ÿ]{2})(?=[A-Z][a-zà-ÿ])", " ", t)
    if not garder_tons:
        t = sans_tons(t)
    return re.sub(r"\s+", " ", t).strip()


def nettoyer_fr(t):
    t = unicodedata.normalize("NFC", t).replace("\xa0", " ").replace(" ", " ")
    t = t.replace("\r", " ").replace("\n", " ").replace("’", "'")
    t = t.replace("[", "").replace("]", "")
    t = re.sub(r"\s+([,.;:!?»])", r" \1", t)  # garde le style français
    return re.sub(r"\s+", " ", t).strip()


# ---------------------------------------------------------------------------
# Passages
# ---------------------------------------------------------------------------

def faire_passages(cles, rn, fr, cible=600, max_versets=8):
    """
    Regroupe des versets qui se suivent (même livre, même chapitre) en
    passages d'environ `cible` caractères de kirundi. Un passage ne garde
    que des versets présents DES DEUX CÔTÉS, et il est coupé dès qu'un
    verset manque (pour ne jamais mélanger le sens).
    """
    passages, courant = [], []

    def fermer():
        if courant:
            b, c = courant[0][0], courant[0][1]
            ref = f"{b} {c},{courant[0][2]}" + (f"-{courant[-1][2]}" if len(courant) > 1 else "")
            passages.append({"ref": ref,
                             "rn": " ".join(rn[k] for k in courant),
                             "fr": " ".join(fr[k] for k in courant)})
        courant.clear()

    precedent = None
    for k in cles:
        ok = k in rn and k in fr and rn[k] and fr[k]
        meme_chapitre = precedent and precedent[:2] == k[:2]
        if not ok or not meme_chapitre:
            fermer()
        if ok:
            courant.append(k)
            if sum(len(rn[x]) for x in courant) >= cible or len(courant) >= max_versets:
                fermer()
        precedent = k
    fermer()
    return passages


MOT = re.compile(r"[A-Za-zÀ-ÿ]+(?:'[A-Za-zÀ-ÿ]+)*")


def ecrire_mots(chemin, versets):
    compte, exemple = collections.Counter(), {}
    for (b, c, v), t in versets.items():
        for m in MOT.findall(t):
            w = m.lower()
            compte[w] += 1
            exemple.setdefault(w, f"{b} {c},{v}")
    with open(chemin, "w", encoding="utf-8") as f:
        f.write("mot\tnombre\tpremière référence\n")
        for w, n in compte.most_common():
            f.write(f"{w}\t{n}\t{exemple[w]}\n")
    return len(compte)


# ---------------------------------------------------------------------------
# Programme principal
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rn", required=True)
    p.add_argument("--fr", required=True)
    p.add_argument("--sortie", required=True)
    p.add_argument("--sft")
    p.add_argument("--part", type=float, default=0.25,
                   help="part des passages gardée pour l'entraînement (0.25 = 25 %%)")
    p.add_argument("--garder_tons", action="store_true")
    p.add_argument("--graine", type=int, default=42)
    args = p.parse_args()
    hasard = random.Random(args.graine)
    os.makedirs(args.sortie, exist_ok=True)

    lignes_rn, titres_rn = lire_db(args.rn)
    lignes_fr, titres_fr = lire_db(args.fr)

    rn = collections.OrderedDict()
    for b, c, v, t in lignes_rn:
        t = nettoyer_rn(t, args.garder_tons)
        if t:
            rn[(b, c, v)] = t
    fr = collections.OrderedDict()
    for b, c, v, t in lignes_fr:
        t = nettoyer_fr(t)
        if t:
            fr[(b, c, v)] = t

    # Ordre de lecture : celui de la Bible kirundi, puis les versets
    # présents seulement en français
    cles = list(rn.keys()) + [k for k in fr if k not in rn]
    passages = faire_passages(cles, rn, fr)

    # Contrôle : un passage dont les longueurs sont très différentes est
    # suspect (verset manquant ou fusionné d'un côté). On le signale et
    # on ne le met pas dans la part d'entraînement.
    suspects = []
    bons = []
    for pa in passages:
        r = len(pa["rn"]) / max(len(pa["fr"]), 1)
        (bons if 0.6 <= r <= 2.0 else suspects).append(pa)

    # Part d'entraînement : régulièrement répartie dans tous les livres
    pas = max(1, round(1 / args.part)) if args.part > 0 else 0
    part = bons[::pas] if pas else []

    # --- Fichiers -------------------------------------------------------------
    def ecrire_texte(nom, cle):
        par_chap = collections.OrderedDict()
        for pa in passages:
            par_chap.setdefault(pa["ref"].split(",")[0], []).append(pa[cle])
        with open(os.path.join(args.sortie, nom), "w", encoding="utf-8") as f:
            f.write("\n\n".join(" ".join(v) for v in par_chap.values()) + "\n")

    # Texte complet (tous les versets, même non alignés) pour chaque langue
    for nom, versets in (("bible_rn.txt", rn), ("bible_fr.txt", fr)):
        par_chap = collections.OrderedDict()
        for (b, c, v), t in versets.items():
            par_chap.setdefault((b, c), []).append(t)
        with open(os.path.join(args.sortie, nom), "w", encoding="utf-8") as f:
            f.write("\n\n".join(" ".join(ts) for ts in par_chap.values()) + "\n")

    with open(os.path.join(args.sortie, "bible_passages.jsonl"), "w", encoding="utf-8") as f:
        for pa in passages:
            f.write(json.dumps(pa, ensure_ascii=False) + "\n")

    blocs = []
    for pa in part:
        if hasard.random() < 0.5:
            blocs.append(f"Kirundi ({pa['ref']}) : {pa['rn']}\nFrançais : {pa['fr']}")
        else:
            blocs.append(f"Français ({pa['ref']}) : {pa['fr']}\nKirundi : {pa['rn']}")
    with open(os.path.join(args.sortie, "bible_bilingue_25.txt"), "w", encoding="utf-8") as f:
        f.write("\n\n".join(blocs) + "\n")

    if args.sft:
        os.makedirs(os.path.dirname(args.sft) or ".", exist_ok=True)
        with open(args.sft, "w", encoding="utf-8") as f:
            for pa in part:
                if hasard.random() < 0.5:
                    q = {"user": f"Traduis en français : {pa['rn']}", "assistant": pa["fr"]}
                else:
                    q = {"user": f"Traduis en kirundi : {pa['fr']}", "assistant": pa["rn"]}
                f.write(json.dumps(q, ensure_ascii=False) + "\n")

    n_mots_rn = ecrire_mots(os.path.join(args.sortie, "mots_bible_rn.tsv"), rn)
    n_mots_fr = ecrire_mots(os.path.join(args.sortie, "mots_bible_fr.tsv"), fr)

    # --- Rapport --------------------------------------------------------------
    alignes = sum(1 for k in rn if k in fr)
    seul_rn = [k for k in rn if k not in fr]
    seul_fr = [k for k in fr if k not in rn]
    lignes = [
        f"Versets kirundi        : {len(rn)}",
        f"Versets français       : {len(fr)}",
        f"Versets alignés        : {alignes}",
        f"Seulement en kirundi   : {len(seul_rn)}  ex. {seul_rn[:5]}",
        f"Seulement en français  : {len(seul_fr)}  ex. {seul_fr[:5]}",
        f"Passages alignés       : {len(passages)}  (suspects : {len(suspects)})",
        f"Part d'entraînement    : {len(part)} passages ({args.part:.0%} des bons)",
        f"Mots différents        : kirundi {n_mots_rn}, français {n_mots_fr}",
        "", "Passages suspects (longueurs très différentes), à vérifier :",
        *[f"  {pa['ref']}: rn {len(pa['rn'])} car. / fr {len(pa['fr'])} car." for pa in suspects[:25]],
        "", "Exemples :",
    ]
    for pa in hasard.sample(bons, min(4, len(bons))):
        lignes += [f"[{pa['ref']}]", f"  rn : {pa['rn'][:300]}", f"  fr : {pa['fr'][:300]}"]
    with open(os.path.join(args.sortie, "rapport.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lignes) + "\n")
    print("\n".join(lignes))


if __name__ == "__main__":
    main()
