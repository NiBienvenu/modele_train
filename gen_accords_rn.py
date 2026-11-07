#!/usr/bin/env python
"""
gen_accords_rn.py : apprend au modèle les ACCORDS du kirundi
(les classes nominales).

L'IDÉE
  En kirundi, chaque nom appartient à une "classe". La classe décide du
  préfixe de TOUS les mots qui s'accordent avec lui :

    umwana mwiza   uyu mwana   umwana wanje   Umwana arakora.   (classe 1)
    abana beza     aba bana    abana banje    Abana barakora.   (classe 2)
    igitabo ciza   iki gitabo  igitabo canje  Igitabo kirazimira. (classe 7)
    ibitabo vyiza  ibi bitabo  ibitabo vyanje Ibitabo birazimira. (classe 8)

  Si le modèle voit des milliers de ces exemples, il apprend la règle :
  "le préfixe du nom commande les autres mots".

D'OÙ VIENNENT LES RÈGLES ?
  Cours FSI "Kirundi Basic Course" (domaine public) : tableaux des
  Unités 1 et 3, vérifiés sur les phrases du livre. Changements de sons
  expliqués par le livre :
    - i ou u + voyelle -> y ou w : mi + iza -> myiza, mu + iza -> mwiza
    - a + i -> e : ba + iza -> beza, ma + iza -> meza
    - ki + voyelle -> c : ciza, coroshe ;  bi + voyelle -> vy : vyiza
    - loi de Dahl : ki -> gi, ka -> ga, tu -> du devant p t k c f s h
      (igiti gito, akantu gato, utuntu duto)
    - classes 9 et 10 : n + adjectif (nziza, nke, nto)

Utilisation dans Colab :
    !python gen_accords_rn.py \
        --noms data/noms_rn.tsv --verbes data/verbes_rn.tsv \
        --sortie /content/drive/MyDrive/donnees/grammaire_rn \
        --sft /content/drive/MyDrive/donnees/qr/accords_rn.jsonl
"""

import argparse
import json
import os
import random

from gen_conjugaison_rn import SOURDES, VOYELLES, avec_ra, avec_zo, lire_verbes, n_plus

# ---------------------------------------------------------------------------
# Le tableau des classes (cours FSI, Unités 1 et 3)
# ---------------------------------------------------------------------------
# sp    : préfixe sujet du verbe (devant consonne)   arakora / kirakora
# sp_v  : préfixe sujet devant voyelle               yandika / candika
# passe : préfixe sujet + a (passé)                  yakoze / cazimiye
# adj   : préfixe d'adjectif devant consonne         munini / kinini
# adj_v : préfixe d'adjectif devant i-               mwiza / ciza
# gen   : "de" / possessif                           wa, ba, ca... -> canje
# ce    : démonstratif "ce, ces" (avant le nom)      uyu mwana, iki gitabo
# cela  : démonstratif "ce ... là"                   uwo mwana, ico gitabo
# None = forme nasale (classes 9 et 10 : nziza, nto)
CLASSES = {
    1:  dict(sp="a",  sp_v="y",  passe="ya",  adj="mu", adj_v="mw", gen="wa",  ce="uyu", cela="uyo"),
    2:  dict(sp="ba", sp_v="b",  passe="ba",  adj="ba", adj_v="be", gen="ba",  ce="aba", cela="abo"),
    3:  dict(sp="u",  sp_v="w",  passe="wa",  adj="mu", adj_v="mw", gen="wa",  ce="uyu", cela="uwo"),
    4:  dict(sp="i",  sp_v="y",  passe="ya",  adj="mi", adj_v="my", gen="ya",  ce="iyi", cela="iyo"),
    5:  dict(sp="ri", sp_v="ry", passe="rya", adj="ri", adj_v="ry", gen="rya", ce="iri", cela="iryo"),
    6:  dict(sp="a",  sp_v="y",  passe="ya",  adj="ma", adj_v="me", gen="ya",  ce="aya", cela="ayo"),
    7:  dict(sp="ki", sp_v="c",  passe="ca",  adj="ki", adj_v="c",  gen="ca",  ce="iki", cela="ico"),
    8:  dict(sp="bi", sp_v="vy", passe="vya", adj="bi", adj_v="vy", gen="vya", ce="ibi", cela="ivyo"),
    9:  dict(sp="i",  sp_v="y",  passe="ya",  adj=None, adj_v=None, gen="ya",  ce="iyi", cela="iyo"),
    10: dict(sp="zi", sp_v="z",  passe="za",  adj=None, adj_v=None, gen="za",  ce="izi", cela="izo"),
    11: dict(sp="ru", sp_v="rw", passe="rwa", adj="ru", adj_v="rw", gen="rwa", ce="uru", cela="urwo"),
    12: dict(sp="ka", sp_v="k",  passe="ka",  adj="ka", adj_v="ke", gen="ka",  ce="aka", cela="ako"),
    13: dict(sp="tu", sp_v="tw", passe="twa", adj="tu", adj_v="tw", gen="twa", ce="utu", cela="utwo"),
    14: dict(sp="bu", sp_v="bw", passe="bwa", adj="bu", adj_v="bw", gen="bwa", ce="ubu", cela="ubwo"),
}

# Préfixes touchés par la loi de Dahl : ki -> gi, ka -> ga, tu -> du
DAHL = {"ki": "gi", "ka": "ga", "tu": "du"}

# Adjectifs : (racine, sens, seulement au pluriel ?)
ADJECTIFS = [
    ("iza", "bon, beau", False),
    ("nini", "grand", False),
    ("to", "petit", False),
    ("inshi", "nombreux", True),
    ("ke", "peu nombreux", True),
]
# Classes 9 et 10 devant voyelle : formes données par le livre / l'usage
NASAL_VOYELLE = {"iza": "nziza", "inshi": "nyinshi"}

# Possessifs : gen + terminaison (wa + nje -> wanje, ca + we -> cawe)
# -we = TON (pas "son") ; -iwe = SON/SA pour une personne ; -nyu = VOTRE
# (Cox Livre I leçons 3 et 60)
POSSESSIFS = [("nje", "mon"), ("we", "ton"), ("iwe", "son"), ("cu", "notre"),
              ("nyu", "votre"), ("bo", "leur")]

# Verbes qui ont du sens selon le type de nom
VERBES_ANIMAL = {"rya", "nywa", "genda", "za", "ryama", "pfa", "rwara", "garuka", "zimira"}
VERBES_CHOSE = {"zimira"}
VERBES_EXCLUS_PERSONNE = {"vyara"}


# ---------------------------------------------------------------------------
# Les règles d'accord
# ---------------------------------------------------------------------------

def adjectif(classe, racine):
    """umwana + iza -> mwiza ; igiti + to -> gito ; inzu + iza -> nziza"""
    c = CLASSES[classe]
    if c["adj"] is None:                          # classes 9 et 10
        if racine[0] in VOYELLES:
            return NASAL_VOYELLE.get(racine)
        return n_plus(racine)                     # nto, nke, nini
    if racine[0] in VOYELLES:
        return c["adj_v"] + racine[1:] if c["adj_v"].endswith("e") else c["adj_v"] + racine
    prefixe = c["adj"]
    if prefixe in DAHL and racine.startswith(SOURDES):
        prefixe = DAHL[prefixe]                   # loi de Dahl
    return prefixe + racine


def possessif(classe, fin):
    """igitabo + nje -> canje ; abana + we -> bawe ; igitabo + iwe -> ciwe"""
    gen = CLASSES[classe]["gen"]
    return gen[:-1] + fin if fin[0] in "aeiou" else gen + fin


def sans_augment(nom):
    """
    Après un démonstratif, le nom perd sa première voyelle :
      umwana -> uyu mwana ; igitabo -> iki gitabo ; inzu -> iyi nzu
    """
    return nom[1:]


def verbe(classe, temps, v):
    """Accorde le verbe avec la classe du nom sujet."""
    c = CLASSES[classe]
    racine, perf = v["racine"], v["perf"]
    if temps == "présent":
        return c["sp"] + avec_ra(racine)
    if temps == "futur":
        zo = avec_zo(racine)
        return c["sp"] + zo if zo else None
    if temps == "passé":
        if perf[0] in VOYELLES:
            return c["sp_v"] + perf            # yize, bize, canditse
        return c["passe"] + perf               # yakoze, cazimiye
    raise ValueError(temps)


# ---------------------------------------------------------------------------
# Lecture des noms
# ---------------------------------------------------------------------------

def lire_noms(chemin):
    noms = []
    with open(chemin, encoding="utf-8") as f:
        for ligne in f:
            if not ligne.strip() or ligne.startswith("#"):
                continue
            c = ligne.rstrip("\n").split("\t")
            if len(c) < 7:
                print(f"  ligne ignorée : {ligne.strip()}")
                continue
            for mot, cl, fr, pluriel in ((c[0], c[1], c[4], False), (c[2], c[3], c[5], True)):
                if mot != "-" and cl.isdigit() and int(cl) in CLASSES:
                    noms.append({"mot": mot, "classe": int(cl), "fr": fr,
                                 "type": c[6], "pluriel": pluriel})
    return noms


def verbes_pour(nom, verbes):
    if nom["type"] == "personne":
        return [v for v in verbes if v["racine"] not in VERBES_EXCLUS_PERSONNE]
    if nom["type"] == "animal":
        return [v for v in verbes if v["racine"] in VERBES_ANIMAL]
    return [v for v in verbes if v["racine"] in VERBES_CHOSE]


# ---------------------------------------------------------------------------
# Textes d'entraînement
# ---------------------------------------------------------------------------

def majuscule(s):
    return s[:1].upper() + s[1:]


def textes(nom, verbes, hasard):
    mot, cl = nom["mot"], nom["classe"]
    c = CLASSES[cl]
    par, qr = [], []

    # 1) Tableau d'accord (bilingue)
    adjs = [(r, s, adjectif(cl, r)) for r, s, plur in ADJECTIFS
            if (nom["pluriel"] or not plur) and adjectif(cl, r)]
    poss = [(f, s, possessif(cl, f)) for f, s in POSSESSIFS]
    lignes = [f"Le nom {mot} ({nom['fr']}) est de la classe {cl}." if nom["fr"]
              else f"Le nom {mot} est de la classe {cl}."]
    lignes += [f"{mot} {a} ({s})" for _, s, a in adjs]
    lignes += [f"{mot} {p} ({s})" for _, s, p in poss]
    lignes += [f"{c['ce']} {sans_augment(mot)} (ce)", f"{c['cela']} {sans_augment(mot)} (ce ... là)"]
    par.append(" ; ".join(lignes) + ".")

    # 2) Phrases en kirundi seul
    phrases = []
    for _, _, a in adjs:
        phrases.append(f"{majuscule(mot)} {a}.")
        phrases.append(f"{majuscule(mot)} ni {a}.")
        phrases.append(f"{majuscule(c['ce'])} {sans_augment(mot)} si {a}.")
    for _, _, p in poss:
        phrases.append(f"{majuscule(mot)} {p}.")
    vs = verbes_pour(nom, verbes)
    for temps, mot_temps in (("présent", "Ubu"), ("passé", "Ejo"), ("futur", "Ejo")):
        if vs:
            v = hasard.choice(vs)
            f = verbe(cl, temps, v)
            if f:
                phrases.append(f"{mot_temps} {mot} {f}.")
    hasard.shuffle(phrases)
    par.append(" ".join(phrases))

    # 3) Questions / réponses
    if adjs and nom["fr"]:
        r, s, a = adjs[0]
        qr.append({"user": f"Comment dit-on « {nom['fr']} » + « {s} » en kirundi ?",
                   "assistant": f"{mot} {a}."})
    qr.append({"user": f"De quelle classe est le nom {mot} ?",
               "assistant": f"{majuscule(mot)} est de la classe {cl}. Exemple : "
                            f"{c['ce']} {sans_augment(mot)}, {mot} {possessif(cl, 'nje')}."})
    if nom["fr"]:
        qr.append({"user": f"Que veut dire {mot} ?",
                   "assistant": f"{majuscule(mot)} veut dire « {nom['fr']} »."})
    if vs:
        v = hasard.choice(vs)
        qr.append({"user": f"Accorde le verbe {v['inf']} au présent avec le sujet {mot}.",
                   "assistant": f"{majuscule(mot)} {verbe(cl, 'présent', v)}."})
    return par, qr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--noms", action="append",
                   help="fichier(s) de noms ; par défaut data/noms_rn.tsv et data/noms_cox.tsv")
    p.add_argument("--verbes", default="data/verbes_rn.tsv")
    p.add_argument("--sortie", required=True)
    p.add_argument("--sft")
    p.add_argument("--repeter", type=int, default=10)
    p.add_argument("--graine", type=int, default=42)
    args = p.parse_args()

    hasard = random.Random(args.graine)
    fichiers = args.noms or [f for f in ("data/noms_rn.tsv", "data/noms_cox.tsv") if os.path.exists(f)]
    noms = [n for f in fichiers for n in lire_noms(f)]
    verbes = lire_verbes(args.verbes)
    print(f"{len(noms)} noms, {len(verbes)} verbes")

    tous_par, tous_qr = [], []
    for k in range(args.repeter):
        for nom in noms:
            # Les noms de Cox sont nombreux : on les répète moins (2 fois)
            if k >= 2 and not nom["fr"]:
                continue
            par, qr = textes(nom, verbes, hasard)
            # Le tableau est le même à chaque fois : gardé une seule fois.
            tous_par.extend(par if k == 0 else par[1:])
            tous_qr.extend(qr if k == 0 else qr[-1:])

    hasard.shuffle(tous_par)
    os.makedirs(args.sortie, exist_ok=True)
    chemin = os.path.join(args.sortie, "accords_rn_001.txt")
    with open(chemin, "w", encoding="utf-8") as f:
        f.write("\n\n".join(tous_par) + "\n")
    print(f"{len(tous_par)} paragraphes écrits dans {chemin}")

    if args.sft:
        os.makedirs(os.path.dirname(args.sft) or ".", exist_ok=True)
        vus = set()
        with open(args.sft, "w", encoding="utf-8") as f:
            for q in tous_qr:
                cle = (q["user"], q["assistant"])
                if cle not in vus:
                    vus.add(cle)
                    f.write(json.dumps(q, ensure_ascii=False) + "\n")
        print(f"{len(vus)} questions/réponses écrites dans {args.sft}")

    # Tableau à vérifier à l'oeil : un nom par classe
    print("\nAccords à vérifier :")
    vus = set()
    for nom in noms:
        if nom["classe"] in vus:
            continue
        vus.add(nom["classe"])
        cl, mot = nom["classe"], nom["mot"]
        adjs = ", ".join(adjectif(cl, r) or "-" for r, _, _ in ADJECTIFS)
        vs = verbes_pour(nom, verbes)
        v = vs[0] if vs else None
        verbes_txt = (f"{verbe(cl, 'présent', v)} / {verbe(cl, 'passé', v)} / "
                      f"{verbe(cl, 'futur', v)}") if v else "-"
        print(f"  cl.{cl:<2} {mot:12s} adj(iza,nini,to,inshi,ke): {adjs:38s} "
              f"{possessif(cl, 'nje'):7s} {CLASSES[cl]['ce']} {sans_augment(mot):11s} {verbes_txt}")


if __name__ == "__main__":
    main()
