"""Repère le kinyarwanda caché dans du texte présenté comme kirundi.

Règles tirées de Zorc & Nibagwire, « Kinyarwanda and Kirundi Comparative
Grammar » (2007), chapitre 1.1 :
  - orthographe (tableau 1) : rw «cy, jy, shy, by»  -> ru «c, j, sh, vy»
  - mots différents (tableaux 2, 4, 6) : cyangwa/canke, icyumweru/indwi, umunsi/umusi...
  - grammaire (tableau 7) : -fite/-fise, futur -zaa-/-zoo-
Règles écartées car fausses alertes dans la Bible kirundi : «ngo», le suffixe -aga
(ikiyaga, umuyaga), et les mots du tableau 3 (le livre dit : « fréquence, pas absence »).

Usage :
    python3 collecte/filtre_kinyarwanda.py fichier.txt            # bilan
    python3 collecte/filtre_kinyarwanda.py fichier.txt --montrer  # + phrases suspectes
Dans un autre script :
    from collecte.filtre_kinyarwanda import est_kinyarwanda
"""

import re
import sys

# --- Indices KINYARWANDA ----------------------------------------------------------
# Orthographe : automatique d'après le livre (Rw «by» = Ru «vy», etc.)
RW_LETTRES = re.compile(r"cy|jy|shy|(?<![a-z])by|[aeiou]by|mby|nby", re.I)
RW_MOTS = {
    # tableau 6 (mots complètement différents)
    "cyangwa", "icyumweru", "umubare", "umusaruro", "umuhinzi", "amacandwe",
    "mutarama", "gashyantare", "werurwe", "gicurasi", "kamena", "kanama", "nzeri",
    "ukwakira", "ugushyingo", "ukuboza",
    # tableau 2 (petites différences)
    "amagambo", "ijosi", "umugati", "nyirabukwe", "nyirakuru", "impyisi",
}
RW_GRAMMAIRE = re.compile(
    r"\b\w*fite\b"            # -fite 'avoir'  (ru -fise)
    r"|\b\w*zaa\w*\b",         # futur -zaa-   (ru -zoo-)
    re.I,
)

# --- Indices KIRUNDI ------------------------------------------------------------------
RU_LETTRES = re.compile(r"vy|(?<![a-z])c[aeiou]|[aeiou]c[aeiou]", re.I)  # ivyo, cane, ico (rw : ibyo, cyane, icyo)
RU_MOTS = {
    "canke", "indwi", "umusi", "imisi", "amajambo", "izosi", "umukate", "inabukwe", "inakuru",
    "umudoka", "umuduga", "impene", "ikinezero", "mugabo", "amahera", "igitigiri", "umwimbu",
    "nzero", "ntwarante", "ndamukiza", "rusama", "ruheshi", "mukakaro", "myandagaro",
    "gitugutu", "munyonyo", "kugira",
}
RU_GRAMMAIRE = re.compile(r"\b\w*fise\b|\b\w*zoo\w*\b", re.I)

MOT = re.compile(r"[a-zA-Zàâéèêîôûç']+")


def scores(phrase: str) -> tuple[int, int]:
    """(points kinyarwanda, points kirundi) d'une phrase."""
    # Les noms propres (Cyprien, Byumba...) ne comptent pas
    bas = " ".join(m for m in phrase.split() if not m[:1].isupper()).lower()
    mots = MOT.findall(bas)
    rw = len(RW_LETTRES.findall(bas)) + sum(m in RW_MOTS for m in mots) + len(RW_GRAMMAIRE.findall(bas))
    ru = len(RU_LETTRES.findall(bas)) + sum(m in RU_MOTS for m in mots) + len(RU_GRAMMAIRE.findall(bas))
    return rw, ru


def est_kinyarwanda(phrase: str) -> bool:
    """Vrai si la phrase a plus d'indices kinyarwanda que kirundi (au moins 1)."""
    rw, ru = scores(phrase)
    return rw >= 1 and rw > ru


if __name__ == "__main__":
    chemin = sys.argv[1]
    montrer = "--montrer" in sys.argv
    total = suspectes = 0
    with open(chemin, encoding="utf-8") as f:
        for ligne in f:
            ligne = ligne.strip()
            if not ligne:
                continue
            total += 1
            if est_kinyarwanda(ligne):
                suspectes += 1
                if montrer and suspectes <= 30:
                    print("  RW?", scores(ligne), ligne[:160])
    print(f"{chemin}: {suspectes}/{total} lignes suspectes ({100 * suspectes / max(total, 1):.1f} %)")
