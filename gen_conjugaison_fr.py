#!/usr/bin/env python
"""
gen_conjugaison_fr.py : crée automatiquement des milliers de conjugaisons
françaises pour la phase A.

D'où viennent les conjugaisons ?
  Du dictionnaire Verbiste, inclus dans la bibliothèque mlconjug3 :
  7 015 verbes vérifiés à la main. On n'utilise PAS la partie
  "intelligence artificielle" de mlconjug3 : elle devine les verbes
  inconnus et peut se tromper (et elle plante avec les nouvelles
  versions de scikit-learn). Le modèle doit apprendre des formes JUSTES.

Ce que le script produit :
  1) Des fichiers .txt pour la phase A (pré-entraînement), sous 3 formes :
     - tableaux :   "Le verbe manger au futur : je mangerai, tu mangeras..."
     - phrases :    "Demain, nous mangerons."
     - participes : "manger : participe présent mangeant, participe passé mangé"
  2) Un fichier .jsonl de questions/réponses pour la phase D :
     {"user": "Conjugue manger au futur.", "assistant": "je mangerai, ..."}

Les temps composés (passé composé, plus-que-parfait...) sont construits
ici avec "avoir" ou "être" + participe passé, avec l'accord quand
l'auxiliaire est "être" (elle est allée, ils sont allés).

Utilisation dans Colab :
    !python gen_conjugaison_fr.py \
        --sortie /content/drive/MyDrive/donnees/conjugaison_fr \
        --sft /content/drive/MyDrive/donnees/qr/conjugaison_fr.jsonl
"""

import argparse
import json
import os
import random
import warnings

warnings.filterwarnings("ignore")  # mlconjug3 affiche des alertes inutiles ici
from mlconjug3.conjug_manager.conjug_manager import ConjugManager  # noqa: E402


# ---------------------------------------------------------------------------
# Les règles fixes du français
# ---------------------------------------------------------------------------

# Les 8 sujets. Le numéro dit quelle forme du verbe utiliser (0 = 1re pers.
# du singulier ... 5 = 3e pers. du pluriel). "accord" dit quelle forme du
# participe passé prendre avec "être" : 0 = masc. sing., 1 = masc. plur.,
# 2 = fém. sing., 3 = fém. plur.
SUJETS = [
    ("je", 0, 0), ("tu", 1, 0), ("il", 2, 0), ("elle", 2, 2),
    ("nous", 3, 1), ("vous", 4, 1), ("ils", 5, 1), ("elles", 5, 3),
]

# Les temps simples : (mode, temps dans Verbiste, nom en français, mot de temps)
TEMPS_SIMPLES = [
    ("Indicatif", "Présent", "présent de l'indicatif", "Aujourd'hui"),
    ("Indicatif", "Imparfait", "imparfait de l'indicatif", "Autrefois"),
    ("Indicatif", "Passé Simple", "passé simple", "Ce jour-là"),
    ("Indicatif", "Futur", "futur simple", "Demain"),
    ("Conditionnel", "Présent", "conditionnel présent", "Avec plus de temps"),
    ("Subjonctif", "Présent", "subjonctif présent", "Il faut que"),
    ("Subjonctif", "Imparfait", "subjonctif imparfait", "Il fallait que"),
]

# Les temps composés : (temps de l'auxiliaire, nom, mot de temps)
TEMPS_COMPOSES = [
    (("Indicatif", "Présent"), "passé composé", "Hier"),
    (("Indicatif", "Imparfait"), "plus-que-parfait", "Avant cela"),
    (("Indicatif", "Futur"), "futur antérieur", "D'ici demain"),
    (("Conditionnel", "Présent"), "conditionnel passé", "Avec plus de temps"),
    (("Subjonctif", "Présent"), "subjonctif passé", "Il faut que"),
]

# Verbes qui se conjuguent avec "être" aux temps composés (quand ils
# n'ont pas de complément d'objet direct).
VERBES_ETRE = {
    "aller", "arriver", "partir", "repartir", "venir", "devenir", "revenir",
    "parvenir", "survenir", "intervenir", "advenir", "entrer", "rentrer",
    "sortir", "ressortir", "naître", "renaître", "mourir", "rester", "tomber",
    "retomber", "monter", "remonter", "descendre", "redescendre", "passer",
    "retourner", "décéder", "apparaître",
}

# Verbes avec un "h aspiré" : on dit "je hais", pas "j'hais".
H_ASPIRE = {
    "haïr", "hacher", "harceler", "hurler", "heurter", "hausser", "hisser",
    "hanter", "hâter", "hennir", "huer", "hérisser", "hasarder", "héler",
    "hiérarchiser", "hocher", "honnir", "houspiller", "humer", "hutter",
}

# Les verbes les plus courants : on les montre plus souvent au modèle,
# comme à l'école où on répète d'abord les verbes de tous les jours.
COURANTS = [
    "être", "avoir", "aller", "faire", "dire", "pouvoir", "vouloir", "savoir",
    "voir", "venir", "devoir", "prendre", "mettre", "donner", "parler",
    "trouver", "penser", "passer", "rester", "partir", "manger", "boire",
    "écrire", "lire", "apprendre", "comprendre", "travailler", "jouer",
    "aimer", "finir", "choisir", "réussir", "grandir", "vivre", "mourir",
    "naître", "ouvrir", "recevoir", "connaître", "croire", "courir",
    "dormir", "sortir", "entrer", "tomber", "acheter", "vendre", "attendre",
    "répondre", "entendre", "appeler", "jeter", "commencer", "payer",
    "envoyer", "cultiver", "construire", "conduire", "servir", "suivre",
]

VOYELLES = tuple("aeiouyéèêëàâäîïôöûüœæh")


# ---------------------------------------------------------------------------
# Lire le dictionnaire Verbiste
# ---------------------------------------------------------------------------

class Conjugueur:
    """Donne les formes d'un verbe à partir du dictionnaire Verbiste."""

    def __init__(self):
        self.m = ConjugManager("fr")
        self.verbes = list(self.m.verbs)

    def modele(self, infinitif):
        info = self.m.get_verb_info(infinitif)
        if info is None:
            return None, None
        return info.root, self.m.get_conjug_info(info.template)

    @staticmethod
    def formes(racine, liste):
        """
        Verbiste range les terminaisons comme [[0, "e"], [1, "es"], ...].
        Forme complète = racine + terminaison. None = la forme n'existe pas
        (exemple : "je pleus" n'existe pas).
        """
        sortie = []
        for _, fin in liste:
            sortie.append(None if fin is None else racine + fin)
        return sortie

    def temps(self, infinitif, mode, temps):
        racine, c = self.modele(infinitif)
        try:
            return self.formes(racine, c[mode][temps])
        except (KeyError, TypeError):
            return None

    def participes(self, infinitif):
        racine, c = self.modele(infinitif)
        p = c["Participe"]
        passe = self.formes(racine, p["Participe Passé"])  # ms, mp, fs, fp
        present = None if p["Participe Présent"] is None else racine + p["Participe Présent"]
        return present, passe

    def imperatif(self, infinitif):
        racine, c = self.modele(infinitif)
        return self.formes(racine, c["Imperatif"]["Imperatif Présent"])


# ---------------------------------------------------------------------------
# Écrire correctement : élisions (j', qu')
# ---------------------------------------------------------------------------

def commence_par_voyelle(mot, infinitif):
    if infinitif in H_ASPIRE:
        return False
    return mot.lower().startswith(VOYELLES)


def avec_sujet(sujet, forme, infinitif, subjonctif=False):
    """
    "je" + "aime" -> "j'aime"   |   "que" + "il" -> "qu'il"
    """
    if sujet == "je" and commence_par_voyelle(forme, infinitif):
        groupe = "j'" + forme
    else:
        groupe = f"{sujet} {forme}"
    if subjonctif:
        groupe = ("qu'" + groupe) if sujet in ("il", "elle", "ils", "elles") else ("que " + groupe)
    return groupe


# ---------------------------------------------------------------------------
# Construire toutes les formes d'un verbe
# ---------------------------------------------------------------------------

def conjuguer(cj, infinitif):
    """
    Retourne un dictionnaire : nom du temps -> liste de (sujet, texte).
    Exemple : "futur simple" -> [("je", "je mangerai"), ("tu", "tu mangeras"), ...]
    """
    resultat = {}

    # 1) Temps simples
    for mode, temps, nom, _ in TEMPS_SIMPLES:
        formes = cj.temps(infinitif, mode, temps)
        if not formes:
            continue
        lignes = []
        for sujet, i, _ in SUJETS:
            if formes[i]:
                lignes.append((sujet, avec_sujet(sujet, formes[i], infinitif,
                                                 subjonctif=(mode == "Subjonctif"))))
        if lignes:
            resultat[nom] = lignes

    # 2) Temps composés : auxiliaire conjugué + participe passé
    _, passe = cj.participes(infinitif)
    aux = "être" if infinitif in VERBES_ETRE else "avoir"
    for (mode, temps), nom, _ in TEMPS_COMPOSES:
        formes_aux = cj.temps(aux, mode, temps)
        if not formes_aux or not passe:
            continue
        lignes = []
        for sujet, i, accord in SUJETS:
            # Avec "avoir" : pas d'accord avec le sujet (elle a mangé).
            # Avec "être" : accord avec le sujet (elle est allée).
            pp = passe[accord] if aux == "être" else passe[0]
            if pp is None or formes_aux[i] is None:
                continue
            # On regarde si le verbe conjugué existe à ce temps simple :
            # "je pleus" n'existe pas, donc "j'ai plu" non plus.
            if cj.temps(infinitif, "Indicatif", "Présent")[i] is None:
                continue
            groupe = avec_sujet(sujet, formes_aux[i], aux, subjonctif=(mode == "Subjonctif"))
            lignes.append((sujet, f"{groupe} {pp}"))
        if lignes:
            resultat[nom] = lignes

    # 3) Impératif (3 formes : tu, nous, vous)
    imp = [f for f in cj.imperatif(infinitif) if f]
    if imp:
        resultat["impératif présent"] = [("", f) for f in imp]

    return resultat, aux


# ---------------------------------------------------------------------------
# Transformer les formes en textes d'entraînement
# ---------------------------------------------------------------------------

MOTS_TEMPS = {nom: mot for _, _, nom, mot in TEMPS_SIMPLES}
MOTS_TEMPS.update({nom: mot for _, nom, mot in TEMPS_COMPOSES})


def textes_pour_verbe(cj, infinitif, hasard):
    """Produit les paragraphes (phase A) et les questions/réponses (phase D)."""
    tout, aux = conjuguer(cj, infinitif)
    if not tout:
        return [], []
    paragraphes, qr = [], []

    # Forme 1 : tableaux, un paragraphe par temps
    for nom, lignes in tout.items():
        liste = ", ".join(t for _, t in lignes)
        paragraphes.append(f"Le verbe {infinitif} au {nom} : {liste}.")
        qr.append({"user": f"Conjugue le verbe {infinitif} au {nom}.",
                   "assistant": f"{liste}."})

    # Forme 2 : petites phrases avec un mot de temps.
    # Le modèle voit le verbe "en situation", comme dans un vrai texte.
    phrases = []
    for nom, lignes in tout.items():
        if nom == "impératif présent":
            continue
        mot = MOTS_TEMPS[nom]
        sujet, texte = hasard.choice(lignes)
        if mot.startswith("Il faut") or mot.startswith("Il fallait"):
            phrases.append(f"{mot} {texte.removeprefix('que ')}.".replace("que qu'", "qu'"))
        else:
            phrases.append(f"{mot}, {texte}.")
    if phrases:
        paragraphes.append(" ".join(phrases))

    # Forme 3 : participes et auxiliaire
    present, passe = cj.participes(infinitif)
    if passe and passe[0]:
        accords = ", ".join(p for p in passe if p)
        paragraphes.append(
            f"Le verbe {infinitif} : participe présent {present}, participe passé "
            f"{passe[0]} ({accords}). Aux temps composés, il se conjugue avec "
            f"l'auxiliaire {aux}.")
        qr.append({"user": f"Quel est le participe passé du verbe {infinitif} ?",
                   "assistant": f"Le participe passé de {infinitif} est {passe[0]}."})
        qr.append({"user": f"Le verbe {infinitif} se conjugue avec quel auxiliaire ?",
                   "assistant": f"Avec l'auxiliaire {aux} : "
                                f"{tout.get('passé composé', [('', '')])[0][1]}."})

    return paragraphes, qr


# ---------------------------------------------------------------------------
# Programme principal
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sortie", required=True, help="dossier des fichiers .txt")
    p.add_argument("--sft", help="fichier .jsonl de questions/réponses (phase D)")
    p.add_argument("--max_verbes", type=int, default=0, help="0 = tous les verbes")
    p.add_argument("--repeter_courants", type=int, default=3,
                   help="combien de fois montrer les verbes courants")
    p.add_argument("--lignes_par_fichier", type=int, default=20000)
    p.add_argument("--graine", type=int, default=42)
    args = p.parse_args()

    hasard = random.Random(args.graine)
    cj = Conjugueur()
    verbes = cj.verbes[: args.max_verbes] if args.max_verbes else cj.verbes
    courants = [v for v in COURANTS if v in cj.m.verbs]
    # Chaque répétition choisit d'autres phrases au hasard : le modèle voit
    # le même verbe dans des situations différentes.
    a_traiter = verbes + courants * args.repeter_courants
    print(f"{len(verbes)} verbes + {len(courants)} verbes courants x {args.repeter_courants}")

    tous_par, tous_qr, erreurs = [], [], 0
    for infinitif in a_traiter:
        try:
            par, qr = textes_pour_verbe(cj, infinitif, hasard)
        except Exception as e:  # un verbe mal rangé dans Verbiste : on l'ignore
            erreurs += 1
            if erreurs <= 5:
                print(f"  ignoré : {infinitif} ({e})")
            continue
        tous_par.extend(par)
        tous_qr.extend(qr)

    # On mélange pour que les fichiers ne soient pas rangés par ordre
    # alphabétique (sinon la fin de l'entraînement ne verrait que des "v").
    hasard.shuffle(tous_par)

    # Fichiers .txt : UNE LIGNE VIDE entre les paragraphes (train.py
    # découpe à cet endroit). "_fr" dans le nom : train_tokenizer.py
    # reconnaît ainsi la langue.
    os.makedirs(args.sortie, exist_ok=True)
    n = args.lignes_par_fichier
    for k in range(0, len(tous_par), n):
        chemin = os.path.join(args.sortie, f"conjugaison_fr_{k // n + 1:03d}.txt")
        with open(chemin, "w", encoding="utf-8") as f:
            f.write("\n\n".join(tous_par[k:k + n]) + "\n")
    print(f"{len(tous_par)} paragraphes écrits dans {args.sortie}")

    if args.sft:
        os.makedirs(os.path.dirname(args.sft) or ".", exist_ok=True)
        hasard.shuffle(tous_qr)
        with open(args.sft, "w", encoding="utf-8") as f:
            for q in tous_qr:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")
        print(f"{len(tous_qr)} questions/réponses écrites dans {args.sft}")

    if erreurs:
        print(f"{erreurs} verbe(s) ignoré(s)")

    # Quelques exemples pour vérifier à l'oeil
    print("\nExemples :")
    for par in tous_par[:6]:
        print(" ", par[:200])


if __name__ == "__main__":
    main()
