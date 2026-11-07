#!/usr/bin/env python
"""
gen_syntaxe_rn.py : phrases kirundi + traduction française pour deux
points de grammaire qui changent le SENS d'une traduction :

1) PRONOMS OBJETS DANS LE VERBE (Cox, Livre I leçons 57-59)
   Le kirundi met l'objet DANS le verbe ; le pronom dépend de la classe
   du nom qu'il reprend. Le français dit seulement le/la/les.
       Urabona igitabo? - Ego, ndakibona.   Tu vois le livre ? - Oui, je le vois.
       Waguze impuzu?   - Ego, narayiguze.  Tu as acheté le vêtement ? - Oui, je l'ai acheté.
       Urankunda?       - Ego, ndagukunda.  Tu m'aimes ? - Oui, je t'aime.
   Règles (Cox) :
     place : après le temps, juste avant la racine (nda-KI-bona)
     classes : 1 mu, 2 ba, 3 wu, 4 yi, 5 ri, 6 ya, 7 ki, 8 bi, 9 yi, 10 zi,
               11 ru, 12 ka, 13 tu, 14 bu ; personnes : n (me), ku (te),
               tu (nous), ba (vous)
     devant voyelle : mu->mw, ba->b, wu->w, yi->y, ri->ry, ya->y, ki->c,
               bi->vy, zi->z, ru->rw, ka->k, tu->tw, bu->bw, ku->kw, n->ny
     loi de Dahl SUR LE PRONOM : ki->gi, ka->ga, tu->du, ku->gu devant
               p t k c f s h (ndagifise) ; le pronom ne change PAS la
               syllabe d'avant (tukibona, pas dukibona)
     n (me) + consonne : n+b->mb, n+h->mp, n+r->nd (arambona, arampa)
     futur + objet : le kw des verbes à voyelle disparaît (nzocibuka)

2) SUBORDONNÉES ET RELATIVES (Cox, Livre I leçons 91-95)
   Pas de -ra- ; négatif avec -ta- (et non nti-/si-) ; -ta- devient -da-
   devant p t k c f s h ; pas de mot pour "qui/que".
       Abana biga barazi gusoma.          Les enfants qui étudient savent lire.
       Abana batiga ntibazi gusoma.       Les enfants qui n'étudient pas ne savent pas lire.
       Nguma i muhira kuko ndwaye.        Je reste à la maison parce que je suis malade.

Utilisation (le français est conjugué avec mlconjug3, comme gen_conjugaison_fr.py) :
    python gen_syntaxe_rn.py --sortie /content/drive/MyDrive/donnees/grammaire_rn \
        --sft /content/drive/MyDrive/donnees/qr/syntaxe_rn.jsonl
"""

import argparse
import json
import os
import random

from gen_conjugaison_fr import Conjugueur, avec_sujet
from gen_conjugaison_rn import SOURDES, VOYELLES, PERSONNES, avec_a_passe, lire_verbes, n_plus, sujet

# ---------------------------------------------------------------------------
# Pronoms objets
# ---------------------------------------------------------------------------
OBJET_CLASSE = {1: "mu", 2: "ba", 3: "wu", 4: "yi", 5: "ri", 6: "ya", 7: "ki", 8: "bi",
                9: "yi", 10: "zi", 11: "ru", 12: "ka", 13: "tu", 14: "bu"}
DEVANT_VOYELLE = {"mu": "mw", "ba": "b", "wu": "w", "yi": "y", "ri": "ry", "ya": "y",
                  "ki": "c", "bi": "vy", "zi": "z", "ru": "rw", "ka": "k", "tu": "tw",
                  "bu": "bw", "ku": "kw"}
DAHL = {"ki": "gi", "ka": "ga", "tu": "du", "ku": "gu"}


def objet_plus(obj, suite):
    """Colle le pronom objet devant la racine (ou le perfectif)."""
    if obj == "n":                       # "me"
        return "ny" + suite if suite[0] in VOYELLES else n_plus(suite)
    if suite[0] in VOYELLES:
        return DEVANT_VOYELLE[obj] + suite
    if obj in DAHL and suite.startswith(SOURDES):
        return DAHL[obj] + suite
    return obj + suite


def sp_sans_dahl(prefixe, suite):
    """Préfixe sujet devant un pronom objet : pas de loi de Dahl (tukibona)."""
    if prefixe == "n":
        return n_plus(suite)
    return prefixe + suite


def present_obj(p, obj, v):          # nda-ki-bona
    return sp_sans_dahl(p, "ra" + objet_plus(obj, v["racine"]))


def passe_obj(p, obj, v):            # na-ki-guze
    debut = {"n": "na", "u": "wa", "a": "ya", "tu": "twa", "mu": "mwa", "ba": "ba"}[p]
    return debut + objet_plus(obj, v["perf"])


def futur_obj(p, obj, v):            # nzo-ki-gura ; nzo-c-ibuka (pas de kw)
    return sp_sans_dahl(p, "zo" + objet_plus(obj, v["racine"]))


def neg_passe_obj(p, obj, v):        # sina-ki-guze, ntiwa-ki-guze
    debut = {"n": "sina", "u": "ntiwa", "a": "ntiya", "tu": "ntitwa", "mu": "ntimwa", "ba": "ntiba"}[p]
    return debut + objet_plus(obj, v["perf"])


# ---------------------------------------------------------------------------
# Français
# ---------------------------------------------------------------------------
# Genre des noms français de data/noms_rn.tsv (m = masculin, f = féminin)
GENRES = {
    "personne": "f", "enfant": "m", "homme": "m", "femme": "f", "enseignant": "m",
    "élève": "m", "travailleur": "m", "domestique": "m", "cuisinier": "m",
    "garçon": "m", "fille": "f", "parent": "m", "roi": "m", "Burundais": "m",
    "voiture": "f", "année": "f", "colline": "f", "orange": "f", "mangue": "f",
    "jour": "m", "arbre": "m", "école": "f", "nom": "m", "oeuf": "m", "mot": "m",
    "lettre": "f", "heure": "f", "stylo": "m", "eau": "f", "chose": "f", "livre": "m",
    "pays": "m", "chaussure": "f", "fruit": "m", "lac": "m", "vache": "f",
    "chien": "m", "maison": "f", "vêtement": "m", "citron": "m", "ananas": "m",
    "chemin": "m", "enclos": "m", "rivière": "f", "feuille de papier": "f",
    "langue": "f", "travail": "m",
}
# Choses qu'on peut acheter, prendre, apporter (pour que les phrases aient du sens)
PORTABLES = {"igitabo", "ibitabo", "ikaramu", "amakaramu", "ikirato", "ibirato", "icamwa",
             "ivyamwa", "umucungwa", "imicungwa", "umwembe", "imyembe", "igi", "amagi",
             "ikete", "amakete", "impuzu", "indimu", "inanasi", "urupapuro", "impapuro",
             "isaha", "amasaha", "amazi"}

PRONOMS_FR = ["je", "tu", "il", "nous", "vous", "ils"]
VOYELLES_FR = tuple("aeiouyéèêàâîôûh")


def article(fr, genre, pluriel):
    if pluriel:
        return "les " + fr
    if fr.startswith(VOYELLES_FR):
        return "l'" + fr
    return ("le " if genre == "m" else "la ") + fr


def pronom_cod(genre, pluriel, verbe_conjugue):
    """le / la / l' / les devant le verbe français."""
    if pluriel:
        return "les "
    if verbe_conjugue.startswith(VOYELLES_FR):
        return "l'"
    return "le " if genre == "m" else "la "


class Francais:
    def __init__(self):
        self.cj = Conjugueur()

    def present(self, inf, i):
        f = self.cj.temps(inf, "Indicatif", "Présent")
        return f[i] if f else None

    def futur(self, inf, i):
        f = self.cj.temps(inf, "Indicatif", "Futur")
        return f[i] if f else None

    def pp(self, inf, genre="m", pluriel=False):
        """participe passé accordé : acheté, achetée, achetés, achetées"""
        _, passe = self.cj.participes(inf)
        idx = {("m", False): 0, ("m", True): 1, ("f", False): 2, ("f", True): 3}[(genre, pluriel)]
        return passe[idx] or passe[0]

    def avoir(self, i):
        return ["ai", "as", "a", "avons", "avez", "ont"][i]


def elider(t):
    """ne + voyelle -> n' ; que + il -> qu'il (français correct)."""
    import re
    t = re.sub(r"\bne ([aeiouéèêh])", r"n'\1", t)
    t = re.sub(r"\bque (il|ils|elle|elles|on)\b", r"qu'\1", t)
    return t


def sujet_fr(i, forme):
    return avec_sujet(PRONOMS_FR[i], forme, "")


# ---------------------------------------------------------------------------
# 1) Dialogues avec pronoms objets
# ---------------------------------------------------------------------------
# racine kirundi -> (infinitif français, objets possibles)
VERBES_OBJET = {
    "bona": ("voir", "tout"), "gura": ("acheter", "portable"), "zana": ("apporter", "portable"),
    "fata": ("prendre", "portable"), "rondera": ("chercher", "tout"), "kunda": ("aimer", "tout"),
}
# Pour "me/te" : verbes qui vont avec une personne
VERBES_PERSONNE = {"kunda": "aimer", "bona": "voir", "fasha": "aider",
                   "rondera": "chercher", "hamagara": "appeler"}


def dialogues_objets(noms, verbes, fr, hasard):
    par, qr = [], []
    v_par_racine = {v["racine"]: v for v in verbes}
    for nom in noms:
        g = GENRES.get(nom["fr"].split(",")[0].strip())
        if not g:
            continue
        nom_fr = nom["fr"].split(",")[0].strip()
        pl = nom["pluriel"]
        obj = OBJET_CLASSE[nom["classe"]]
        for racine, (inf, sorte) in VERBES_OBJET.items():
            v = v_par_racine.get(racine)
            if not v or (sorte == "portable" and nom["mot"] not in PORTABLES):
                continue
            le_nom = article(nom_fr, g, pl)
            tu_pres, je_pres = fr.present(inf, 1), fr.present(inf, 0)
            je_fut = fr.futur(inf, 0)
            if not (tu_pres and je_pres and je_fut):
                continue
            pp = fr.pp(inf, g, pl)
            pp_simple = fr.pp(inf)
            q_pres_rn = f"Ura{v['racine']} {nom['mot']}?"
            q_pres_fr = f"Tu {tu_pres} {le_nom} ?"
            r_pres_rn = f"Ego, {present_obj('n', obj, v)}."
            r_pres_fr = f"Oui, {sujet_fr(0, pronom_cod(g, pl, je_pres) + je_pres)}."
            q_pas_rn = f"{avec_a_passe('u', v['perf']).capitalize()} {nom['mot']}?"
            q_pas_fr = f"Tu as {pp_simple} {le_nom} ?"
            r_pas_rn = f"Ego, {passe_obj('n', obj, v)}."
            r_pas_fr = f"Oui, je {pronom_cod(g, pl, 'ai').strip()}{'' if pronom_cod(g, pl, 'ai') == 'l' + chr(39) else ' '}ai {pp}.".replace("je l' ai", "je l'ai")
            r_neg_rn = f"Oya, {neg_passe_obj('n', obj, v)}."
            r_neg_fr = f"Non, je ne {pronom_cod(g, pl, 'ai').strip()}{'' if pronom_cod(g, pl, 'ai') == 'l' + chr(39) else ' '}ai pas {pp}.".replace("ne l' ai", "ne l'ai")
            r_fut_rn = f"{futur_obj('n', obj, v).capitalize()} ejo."
            r_fut_fr = f"{sujet_fr(0, pronom_cod(g, pl, je_fut) + je_fut).capitalize()} demain."
            par.append(f"{q_pres_rn} {r_pres_rn}\n{q_pres_fr} {r_pres_fr}")
            par.append(f"{q_pas_rn} {r_pas_rn} {r_fut_rn}\n{q_pas_fr} {r_pas_fr} {r_fut_fr}")
            par.append(f"{q_pas_rn} {r_neg_rn}\n{q_pas_fr} {r_neg_fr}")
            qr.append({"user": f"Traduis en kirundi : {q_pres_fr} {r_pres_fr}",
                       "assistant": f"{q_pres_rn} {r_pres_rn}"})
            qr.append({"user": f"Traduis en français : {q_pas_rn} {r_pas_rn}",
                       "assistant": f"{q_pas_fr} {r_pas_fr}"})
            qr.append({"user": f"Réponds en kirundi avec un pronom : {q_pres_rn}",
                       "assistant": f"{r_pres_rn} (le pronom -{obj}- reprend {nom['mot']}, classe {nom['classe']})"})

    # me / te / nous
    for racine, inf in VERBES_PERSONNE.items():
        v = v_par_racine.get(racine)
        if not v:
            continue
        tu_m = fr.present(inf, 1)
        je_t = fr.present(inf, 0)
        if not (tu_m and je_t):
            continue
        q_rn = f"{sp_sans_dahl('u', 'ra' + objet_plus('n', v['racine'])).capitalize()}?"
        r_rn = f"Ego, {present_obj('n', 'ku', v)}."
        me = "m'" if tu_m.startswith(VOYELLES_FR) else "me "
        te = "t'" if je_t.startswith(VOYELLES_FR) else "te "
        q_fr = f"Tu {me}{tu_m} ?".replace("me  ", "me ")
        r_fr = f"Oui, {sujet_fr(0, te + je_t)}."
        il_nous = fr.present(inf, 2)
        a_rn = f"{sp_sans_dahl('a', 'ra' + objet_plus('tu', v['racine'])).capitalize()}."
        a_fr = f"Il nous {il_nous}." if not il_nous.startswith(VOYELLES_FR) else f"Il nous {il_nous}."
        par.append(f"{q_rn} {r_rn} {a_rn}\n{q_fr} {r_fr} {a_fr}")
        qr.append({"user": f"Traduis en kirundi : {q_fr} {r_fr}", "assistant": f"{q_rn} {r_rn}"})
        qr.append({"user": f"Traduis en français : {a_rn}", "assistant": a_fr})
    return par, qr


# ---------------------------------------------------------------------------
# 2) Subordonnées et relatives
# ---------------------------------------------------------------------------
def dep_neg(p, suite):
    """Négatif de subordonnée : sujet + ta (da devant consonne sourde) ; ta+i -> ti."""
    ta = "da" if suite.startswith(SOURDES) else "ta"
    if suite[0] in VOYELLES:
        ta = "t"
    debut = {"n": "n", "u": "u", "a": "a", "tu": "tu", "mu": "mu", "ba": "ba"}[p]
    return debut + ta + suite


# Phrases "parce que" : (principale kirundi par personne, français, raison)
# rwaye = état "être malade" (Cox : statif sans -ra- en subordonnée)
def phrases_kuko(fr):
    par, qr = [], []
    modeles = [
        # (racine principale, complément rn, infinitif fr, complément fr, négatif principal ?, raison)
        ("guma", "i muhira", "rester", "à la maison", False, "malade"),
        ("kora", "uyu musi", "travailler", "aujourd'hui", True, "malade"),
        ("genda", "ku kazi", "aller", "au travail", True, "malade"),
    ]
    for racine, comp_rn, inf, comp_fr, negatif_principal, _ in modeles:
        for i, (_, p, _) in enumerate(PERSONNES):
            if negatif_principal:
                if p == "n":
                    princ = "si" + n_plus(racine)
                else:
                    f = sujet(p, racine)
                    princ = ("nt" + f) if p in ("u", "a") else "nti" + f
                v_fr = fr.present(inf, i)
                if not v_fr:
                    continue
                fr_princ = sujet_fr(i, "ne " + v_fr + " pas").replace("je ne ", "je ne ")
                fr_princ = fr_princ.replace(" ne a", " n'a").replace(" ne e", " n'e").replace(" ne v", " ne v")
            else:
                princ = sujet(p, racine)
                v_fr = fr.present(inf, i)
                if not v_fr:
                    continue
                fr_princ = sujet_fr(i, v_fr)
            raison_rn = sujet(p, "rwaye")
            etre = ["suis", "es", "est", "sommes", "êtes", "sont"][i]
            raison_fr = f"{PRONOMS_FR[i]} {etre} malade" if i != 0 else "je suis malade"
            rn = f"{princ.capitalize()} {comp_rn} kuko {raison_rn}."
            fr_ = f"{fr_princ[0].upper() + fr_princ[1:]} {comp_fr} parce qu{'e ' if raison_fr[0] not in 'aeiouéèêi' else chr(39)}{raison_fr}."
            fr_ = fr_.replace("parce que il", "parce qu'il").replace("parce que ils", "parce qu'ils")
            fr_ = elider(fr_)
            if i >= 3:                       # nous, vous, ils : malades
                fr_ = fr_.replace(" malade.", " malades.")
            par.append(f"Kirundi : {rn}\nFrançais : {fr_}")
            qr.append({"user": f"Traduis en kirundi : {fr_}", "assistant": rn})
            # Négatif de subordonnée : -ta- (et non nti-)
            raison_neg = dep_neg(p, "rwaye")
            rn2 = f"{sujet(p, 'ra' + 'kora').capitalize()} kuko {raison_neg}."
            v_tr = fr.present("travailler", i)
            fr2 = f"{sujet_fr(i, v_tr).capitalize()} parce qu{'e ' if PRONOMS_FR[i][0] not in 'aeiou' else chr(39)}{PRONOMS_FR[i]} ne {'suis' if i == 0 else etre} pas malade."
            fr2 = fr2.replace("parce qu'je", "parce que je").replace("parce que il", "parce qu'il").replace("parce que ils", "parce qu'ils")
            fr2 = elider(fr2)
            if i >= 3:
                fr2 = fr2.replace(" malade.", " malades.")
            par.append(f"Kirundi : {rn2}\nFrançais : {fr2}")
            qr.append({"user": f"Traduis en français : {rn2}", "assistant": fr2})
    return par, qr


def phrases_relatives(noms, fr):
    """Abana biga barazi gusoma / Abana batiga ntibazi gusoma."""
    par, qr = [], []
    personnes = [n for n in noms if n["type"] == "personne" and GENRES.get(n["fr"].split(",")[0].strip())]
    for n in personnes:
        nom_fr = n["fr"].split(",")[0].strip()
        g = GENRES[nom_fr]
        pl = n["pluriel"]
        p = "ba" if pl else "a"
        i = 5 if pl else 2
        le_nom = article(nom_fr, g, pl)
        # verbe de la relative : kwiga (étudier) ; principale : -zi (savoir, verbe défectif)
        rel = sujet(p, "iga")                       # biga / yiga
        rel_neg = dep_neg(p, "iga")                 # batiga / atiga
        zi = "barazi" if pl else "arazi"
        zi_neg = "ntibazi" if pl else "ntazi"
        etudier = fr.present("étudier", i)
        savoir = fr.present("savoir", i)
        rn1 = f"{n['mot'].capitalize()} {rel} {zi} gusoma."
        fr1 = f"{le_nom.capitalize()} qui {etudier} {savoir} lire."
        rn2 = f"{n['mot'].capitalize()} {rel_neg} {zi_neg} gusoma."
        fr2 = f"{le_nom.capitalize()} qui n'{etudier} pas ne {savoir} pas lire."
        for rn, f_ in ((rn1, fr1), (rn2, fr2)):
            par.append(f"Kirundi : {rn}\nFrançais : {f_}")
            qr.append({"user": f"Traduis en français : {rn}", "assistant": f_})
            qr.append({"user": f"Traduis en kirundi : {f_}", "assistant": rn})
    return par, qr


def lire_noms_fr(chemin):
    noms = []
    for ligne in open(chemin, encoding="utf-8"):
        if ligne.startswith("#") or not ligne.strip():
            continue
        c = ligne.rstrip("\n").split("\t")
        if len(c) < 7:
            continue
        for mot, cl, f, pl in ((c[0], c[1], c[4], False), (c[2], c[3], c[5], True)):
            if mot != "-" and f and cl.isdigit() and int(cl) in OBJET_CLASSE:
                noms.append({"mot": mot, "classe": int(cl), "fr": f, "type": c[6], "pluriel": pl})
    return noms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--noms", default="data/noms_rn.tsv")
    p.add_argument("--verbes", default="data/verbes_rn.tsv")
    p.add_argument("--sortie", required=True)
    p.add_argument("--sft")
    p.add_argument("--graine", type=int, default=42)
    args = p.parse_args()
    hasard = random.Random(args.graine)

    noms = lire_noms_fr(args.noms)
    verbes = lire_verbes(args.verbes)
    fr = Francais()
    par1, qr1 = dialogues_objets(noms, verbes, fr, hasard)
    par2, qr2 = phrases_kuko(fr)
    par3, qr3 = phrases_relatives(noms, fr)
    par, qr = par1 + par2 + par3, qr1 + qr2 + qr3
    hasard.shuffle(par)

    os.makedirs(args.sortie, exist_ok=True)
    with open(os.path.join(args.sortie, "syntaxe_rn_001.txt"), "w", encoding="utf-8") as f:
        f.write("\n\n".join(par) + "\n")
    if args.sft:
        os.makedirs(os.path.dirname(args.sft) or ".", exist_ok=True)
        with open(args.sft, "w", encoding="utf-8") as f:
            for q in qr:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"Pronoms objets : {len(par1)} paragraphes ; kuko : {len(par2)} ; relatives : {len(par3)}")
    print(f"Questions/réponses : {len(qr)}")
    print("\nExemples :")
    for x in par1[:4] + par1[-2:] + par2[:4] + par3[:4]:
        print(" ", x.replace("\n", "\n  "))
        print()


if __name__ == "__main__":
    main()
