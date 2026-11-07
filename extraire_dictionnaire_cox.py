#!/usr/bin/env python
"""
extraire_dictionnaire_cox.py : extrait le dictionnaire kirundi-anglais
d'Elizabeth E. Cox (PDF sur 3 colonnes).

Ce que ce dictionnaire nous donne (partie kirundi -> anglais) :
  VERBES : l'infinitif ET la terminaison du perfectif
      ku-bariza (-je) 1. to ask of     -> kubariza, perfectif barije
      gu-kubita (-se) to beat           -> gukubita, perfectif kubise
      ku-barirwa (-we) to be told       -> kubarirwa, perfectif barirwe
  NOMS : le préfixe du singulier ET celui du pluriel (donc la classe)
      umu-banyi (aba-) neighbor         -> umubanyi / ababanyi (classes 1/2)
      i-bavu (ama-) callus              -> ibavu / amabavu (classes 5/6)
      uru-banza (pl. imanza) trial      -> urubanza / imanza (classes 11/10)

COMMENT LIRE "(-je)" ?
  La terminaison remplace la DERNIÈRE SYLLABE de la racine
  (consonnes + "a") :
      bari-za + -je    -> barije
      kubi-ta + -se    -> kubise
      ku-nda  + -nze   -> kunze
  Si la terminaison commence par la dernière lettre de ces consonnes,
  on garde les consonnes d'avant :
      bari-rwa + -we   -> barirwe   (on garde le "r")

Les traits sur les voyelles (ā, ī...) marquent la longueur/le ton : on
les retire, comme dans l'écriture courante.

FICHIERS PRODUITS (--sortie)
  cox_verbes.tsv  : infinitif, racine, perfectif, sens anglais
  cox_noms.tsv    : singulier, classe, pluriel, classe, sens anglais
  cox_autres.tsv  : autres mots (adverbes, conjonctions...), sens anglais
  rapport.txt     : chiffres, exemples, et COMPARAISON avec nos perfectifs
                    (data/verbes_rn.tsv) : chaque désaccord est listé

Utilisation :
    python extraire_dictionnaire_cox.py --pdf kirundi.pdf --sortie data/cox
"""

import argparse
import collections
import os
import re
import unicodedata

import pymupdf

VOYELLES = "aeiou"

# Préfixes de noms : (singulier, pluriel) -> (classe sg, classe pl)
CLASSES = {
    ("umu", "aba"): (1, 2), ("umw", "ab"): (1, 2), ("umu", "imi"): (3, 4),
    ("umw", "imy"): (3, 4), ("i", "ama"): (5, 6), ("iri", "ama"): (5, 6),
    ("ij", "am"): (5, 6), ("iki", "ibi"): (7, 8), ("igi", "ibi"): (7, 8),
    ("ic", "ivy"): (7, 8), ("in", "in"): (9, 10), ("im", "im"): (9, 10),
    ("iny", "iny"): (9, 10), ("uru", "in"): (11, 10), ("uru", "im"): (11, 10),
    ("urw", "inz"): (11, 10), ("aka", "utu"): (12, 13), ("aga", "udu"): (12, 13),
    ("ak", "utw"): (12, 13), ("ubu", "ama"): (14, 6), ("ubw", "am"): (14, 6),
    ("uku", "ama"): (15, 6), ("ukw", "am"): (15, 6), ("aha", "aha"): (16, 16),
}
PREFIXES_VERBE = ("ku", "gu", "kw", "gw")


def sans_tons(t):
    d = unicodedata.normalize("NFD", t)
    d = "".join(c for c in d if unicodedata.category(c) != "Mn")
    return unicodedata.normalize("NFC", d).replace("’", "'")


# ---------------------------------------------------------------------------
# 1. Lire le PDF colonne par colonne
# ---------------------------------------------------------------------------

def lire_lignes(chemin, page_debut, page_fin):
    """Retourne les lignes dans l'ordre de lecture : colonne 1, 2, 3, page après page."""
    doc = pymupdf.open(chemin)
    sortie = []
    for n in range(page_debut - 1, page_fin):
        page = doc[n]
        lignes = []
        for b in page.get_text("dict")["blocks"]:
            for l in b.get("lines", []):
                t = "".join(s["text"] for s in l["spans"]).strip()
                if t:
                    lignes.append((l["bbox"][0], l["bbox"][1], t))
        # Les 3 colonnes commencent vers x = 72, 225 et 378
        def colonne(x):
            return 0 if x < 215 else (1 if x < 370 else 2)
        lignes.sort(key=lambda e: (colonne(e[0]), e[1]))
        for x, y, t in lignes:
            if re.fullmatch(r"\d{1,3}|[A-Z]|Kirundi - English|English - Kirundi", t):
                continue  # numéro de page, lettre de section, titre
            sortie.append(t)
    return sortie


# Un nouvel article commence par un mot avec trait d'union ("ku-bariza")
# ou un mot suivi d'une catégorie ("kuki (adv.)").
DEBUT_ARTICLE = re.compile(
    r"^(?:[a-zA-Zāēīōūâêîôûǎěǐǒǔ'’]{1,4}-[a-zāēīōūâêîôûǎěǐǒǔ'’-]+"
    r"|[a-zāēīōūâêîôûǎěǐǒǔ'’]+ \((?:adv|conj|interj|prep|pron|adj|n|v)\.)")


def articles(lignes):
    courant = None
    for t in lignes:
        if DEBUT_ARTICLE.match(t):
            if courant:
                yield courant
            courant = t
        elif courant:
            courant += " " + t
    if courant:
        yield courant


# ---------------------------------------------------------------------------
# 2. Comprendre un article
# ---------------------------------------------------------------------------

def appliquer_perfectif(racine, fin):
    """
    Lit la notation de Cox "(-terminaison)".
      1) Forme complète (verbes courts/irréguliers) : la terminaison
         commence comme la racine et est presque aussi longue :
            genda (-giye) -> giye ; rya (-riye) -> riye ; pfa (-pfuye) -> pfuye
      2) Sinon elle remplace la dernière syllabe (consonnes + a) :
            bariza (-je) -> barije ; kubita (-se) -> kubise
         - si elle commence par une des consonnes de cette syllabe, on
           coupe à cette consonne :
            rindira (-riye) -> rindiriye ; barirwa (-we) -> barirwe ;
            ririmba (-mvye) -> ririmvye ; kunda (-nze) -> kunze
         - sinon on garde le "m"/"n" du début : temba (-vye) -> temvye
    """
    # (il faut une voyelle à l'intérieur : "giye", "riye", mais pas "tse")
    if fin[0] == racine[0] and len(fin) >= len(racine) - 1 and re.search(r"[aeiou]", fin[:-1]):
        return fin
    m = re.match(r"^(.*?)([^aeiou]*)a$", racine)
    if not m:
        return None
    debut, cons = m.group(1), m.group(2)
    j = cons.find(fin[0]) if cons else -1
    if j >= 0:
        return debut + cons[:j] + fin
    if len(cons) > 1 and cons[0] in "mn":
        return debut + cons[0] + fin
    return debut + fin


def pluriel_de(prefixe_pl, racine):
    """aba + ana -> abana ; imi + aka -> imyaka ; ibi + umba -> ivyumba"""
    if racine and racine[0] in VOYELLES:
        if prefixe_pl.endswith("a"):
            return prefixe_pl[:-1] + racine
        if prefixe_pl == "ibi":
            return "ivy" + racine
        if prefixe_pl.endswith("i") or prefixe_pl.endswith("u"):
            return prefixe_pl[:-1] + {"i": "y", "u": "w"}[prefixe_pl[-1]] + racine
    return prefixe_pl + racine


ARTICLE = re.compile(r"^(?P<pre>[a-z']{1,4})-(?P<rac>[a-z'-]+)\s*(?:\((?P<par>[^)]*)\))?\s*(?P<sens>.*)$")


def analyser(texte):
    """Retourne ("verbe"|"nom"|"autre", données) ou None."""
    brut = sans_tons(texte)
    m = ARTICLE.match(brut)
    if not m:
        m2 = re.match(r"^([a-z']+) \(([a-z]+)\.\)\s*(.*)$", brut)
        if m2:
            return "autre", {"mot": m2.group(1), "cat": m2.group(2), "sens": m2.group(3)}
        return None
    pre, rac, par, sens = m.group("pre"), m.group("rac"), (m.group("par") or ""), m.group("sens").strip()
    mot = pre + rac

    # --- Verbe : ku-/gu-/kw- + racine en -a + "(-terminaison)" ---
    if pre in PREFIXES_VERBE and par.startswith("-") and rac.endswith("a"):
        fin = par[1:].strip().split()[0].strip(",;")
        if not re.fullmatch(r"[a-z]+", fin):
            return None
        perf = appliquer_perfectif(rac, fin)
        if perf:
            return "verbe", {"inf": mot, "racine": rac, "perf": perf, "fin": fin, "sens": sens}

    # --- Nom : préfixe + "(préfixe du pluriel-)" ou "(pl. forme)" ---
    pl, cl = None, None
    mp = re.fullmatch(r"([a-z]+)-", par.strip())
    if mp:
        pl_pre = mp.group(1)
        cl = CLASSES.get((pre, pl_pre))
        pl = pluriel_de(pl_pre, rac) if pl_pre not in ("in", "im", "iny") else pl_pre + rac
    elif par.startswith("pl.") and par[3:].strip():
        pl = par[3:].strip().split()[0].strip(",;")
    elif "no pl" in par:
        pl = "-"
    if pl is not None:
        return "nom", {"sg": mot, "pl": pl, "cl": cl, "sens": sens}
    return "autre", {"mot": mot, "cat": "", "sens": (f"({par}) " if par else "") + sens}


# ---------------------------------------------------------------------------
# 3. Programme principal
# ---------------------------------------------------------------------------

def lire_nos_verbes(chemin):
    nos = {}
    if os.path.exists(chemin):
        for ligne in open(chemin, encoding="utf-8"):
            if ligne.strip() and not ligne.startswith("#"):
                c = ligne.rstrip("\n").split("\t")
                if len(c) >= 3:
                    nos[c[0]] = (c[2], c[4] if len(c) > 4 else "")
    return nos


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pdf", required=True)
    p.add_argument("--sortie", default="data/cox")
    p.add_argument("--pages", default="5-119", help="pages de la partie kirundi -> anglais")
    p.add_argument("--nos_verbes", default="data/verbes_rn.tsv")
    args = p.parse_args()
    os.makedirs(args.sortie, exist_ok=True)

    debut, fin = (int(x) for x in args.pages.split("-"))
    lignes = lire_lignes(args.pdf, debut, fin)
    verbes, noms, autres, rates = {}, {}, {}, []
    for a in articles(lignes):
        r = analyser(a)
        if r is None:
            rates.append(a)
            continue
        genre, d = r
        if genre == "verbe":
            if d["inf"] in verbes:
                # Deux verbes qui s'écrivent pareil (guhura -ye / guhura -ze)
                verbes[d["inf"]].setdefault("autres", []).append(d["perf"])
            else:
                verbes[d["inf"]] = d
        elif genre == "nom":
            noms.setdefault(d["sg"], d)
        else:
            autres.setdefault(d["mot"], d)

    with open(os.path.join(args.sortie, "cox_verbes.tsv"), "w", encoding="utf-8") as f:
        f.write("# infinitif\tracine\tperfectif\tsens (anglais)\tsource\n")
        for v in verbes.values():
            f.write(f"{v['inf']}\t{v['racine']}\t{v['perf']}\t{v['sens']}\tCox\n")
            for autre in v.get("autres", []):   # homonymes (guhura -ye / -ze)
                f.write(f"{v['inf']}\t{v['racine']}\t{autre}\t(homonyme)\tCox\n")
    with open(os.path.join(args.sortie, "cox_noms.tsv"), "w", encoding="utf-8") as f:
        f.write("# singulier\tclasse\tpluriel\tclasse\tsens (anglais)\n")
        for n in noms.values():
            c1, c2 = n["cl"] if n["cl"] else ("?", "?")
            f.write(f"{n['sg']}\t{c1}\t{n['pl']}\t{c2}\t{n['sens']}\n")
    with open(os.path.join(args.sortie, "cox_autres.tsv"), "w", encoding="utf-8") as f:
        for a in autres.values():
            f.write(f"{a['mot']}\t{a['cat']}\t{a['sens']}\n")

    # --- Comparaison avec nos perfectifs ---------------------------------------
    nos = lire_nos_verbes(args.nos_verbes)
    accords, desaccords, absents = [], [], []
    for inf, (perf, src) in nos.items():
        if inf in verbes:
            possibles = [verbes[inf]["perf"]] + verbes[inf].get("autres", [])
            (accords if perf in possibles else desaccords).append(
                (inf, perf, " / ".join(possibles), src))
        else:
            absents.append(inf)

    lignes_r = [
        f"Pages lues : {debut}-{fin}",
        f"Verbes (avec perfectif) : {len(verbes)}",
        f"Noms (avec pluriel)     : {len(noms)}  (classe reconnue : {sum(1 for n in noms.values() if n['cl'])})",
        f"Autres mots             : {len(autres)}",
        f"Articles non compris    : {len(rates)}",
        "",
        f"COMPARAISON avec data/verbes_rn.tsv ({len(nos)} verbes) :",
        f"  mêmes perfectifs : {len(accords)}",
        f"  DÉSACCORDS       : {len(desaccords)}",
        *[f"    {i:16s} nous : {a:14s} Cox : {b:14s} ({s})" for i, a, b, s in desaccords],
        f"  absents de Cox   : {len(absents)}",
        "", "Exemples de verbes :",
        *[f"  {v['inf']} -> {v['perf']}  ({v['sens'][:50]})" for v in list(verbes.values())[:15]],
        "", "Exemples de noms :",
        *[f"  {n['sg']} / {n['pl']}  cl.{n['cl']}  ({n['sens'][:40]})" for n in list(noms.values())[:15]],
        "", "Articles non compris (exemples) :", *[f"  {a[:90]}" for a in rates[:15]],
    ]
    with open(os.path.join(args.sortie, "rapport.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lignes_r) + "\n")
    print("\n".join(lignes_r))


if __name__ == "__main__":
    main()
