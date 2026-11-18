# dataset_un : ce que le modèle UN apprend en plus du modèle zéro

À mettre sur Kaggle comme **nouveau dataset** (nom : `dataset-un`), ou dans Mon Drive pour Colab.
L'ancien dataset (`dataset`) reste tel quel : le modèle un relit les deux.

- `lecture_rn/` : textes kirundi vérifiés (livres scannés recopiés et contrôlés, textes de messe)
- `insiguro/` : sens des mots difficiles (Dusome)
- `dialogues/` : questions-réponses sur les insiguro
- `tests/` : FLORES (traduction FR <-> RN par des professionnels), jamais vu à l'entraînement
- `code/` : code d'entraînement (2 GPU, sauvegarde toutes les 4 h, pause avant la fin de session)
- `notebooks/` : un_kaggle, un_colab, un_machine

Refaire ce dossier : `python3 preparer_donnees_un.py` dans modele_train/.
