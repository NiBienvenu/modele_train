"""
separation.py : décide, une fois pour toutes, quelles données servent aux
TESTS et ne doivent JAMAIS être vues pendant l'entraînement.

La règle est fixe (un calcul sur le texte) : le même mot ou la même
phrase est toujours du même côté, sur n'importe quel ordinateur.
Tous les générateurs l'utilisent pour EXCLURE les éléments de test.
"""

import hashlib


def _hash(texte):
    return int(hashlib.md5(texte.strip().lower().encode("utf-8")).hexdigest(), 16)


def est_test(texte, pourcent=10):
    """True si cet élément est réservé aux tests (pourcent % des éléments)."""
    return _hash(texte) % 100 < pourcent


# Réglages communs
POURCENT_PAIRES = 10     # paires de traduction (cours de Cox) gardées pour les tests
POURCENT_VERBES = 5      # verbes jamais conjugués à l'entraînement


def verbe_de_test(infinitif):
    return est_test("verbe:" + infinitif, POURCENT_VERBES)


def paire_de_test(rn):
    return est_test("paire:" + rn, POURCENT_PAIRES)
