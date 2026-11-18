# dataset : tout ce qu'il faut pour entraîner le modèle ZÉRO (français + kirundi)

Le même dossier marche partout :
- **Colab** : glisse ce dossier dans Mon Drive, ouvre `notebooks/zero_colab.ipynb`.
- **Kaggle** : crée un dataset privé avec ce dossier, ouvre `notebooks/zero_kaggle.ipynb`.
- **Machine** : ouvre `modele_train/zero_machine.ipynb`.

| Dossier | Contenu |
|---|---|
| lecture_rn/ | vrai texte kirundi : presse (KIRNEWS), Wikipédia rn, web (MADLAD), ikirundi.org, Bible |
| lecture_fr/ | Bible en français (Wikipédia FR est téléchargée par le notebook) |
| lecture_bilingue/ | phrases des cours de Cox et lexique, français + kirundi |
| grammaire/ | un petit échantillon des tableaux de conjugaison et d'accords |
| dialogues/ | salutations, petites conversations, traductions (Bible, cours), vocabulaire, titres d'articles |
| code/ | train.py, configs, tests, téléchargement des données publiques |
| notebooks/ | les 3 notebooks |

Aucune phrase kirundi n'est inventée : elles viennent toutes de textes écrits par des humains
(cours et dictionnaire de Cox, Bible, presse, sites). Ne pas publier ce dossier : les textes
appartiennent à leurs auteurs.

Refaire ce dossier : `python3 preparer_donnees_zero.py` dans modele_train/.
