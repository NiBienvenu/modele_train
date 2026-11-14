# modele_train : modèle français + kirundi (base Pythia-14m)

Un petit modèle (environ 122 millions de paramètres) qui comprend, conjugue
et traduit le **français** et le **kirundi**. Il part de `EleutherAI/pythia-14m`,
qu'on fait grandir (`grow.py`) pendant l'entraînement.

Deux façons d'entraîner :
- **Google Colab** (GPU T4 gratuit) : ce guide, ci-dessous ;
- **sur ton ordinateur** (GPU NVIDIA local) : voir [README_LOCAL.md](README_LOCAL.md).

---

## Guide rapide : entraîner le modèle sur Colab

### 1. Mettre les données dans Google Drive (une seule fois)

Sur ton ordinateur, le dossier `A_METTRE_DANS_DRIVE/` contient tout ce qu'il faut
(111 Mo). Il n'est **pas** sur GitHub (trop gros, et sous droits d'auteur).

1. Ouvre https://drive.google.com avec **devbienvenu@gmail.com**.
2. Ouvre `A_METTRE_DANS_DRIVE/` sur ton ordinateur et **glisse ses 3 dossiers**
   (`sources`, `donnees`, `modele`) dans **Mon Drive** (à la racine).
3. Attends la fin de l'envoi. Tu dois voir :
   ```
   Mon Drive/
   ├── sources/   bible_kirundi.db, bible_francais.db
   ├── donnees/   conjugaison_fr, conjugaison_rn, grammaire_rn, bible, cours, lexique_rn, fsi_rn, qr
   └── modele/    tokenizer_fr_rn, journaux
   ```

### 2. Ouvrir le notebook

1. Va sur https://colab.research.google.com (même compte Gmail).
2. **Fichier → Ouvrir un notebook → GitHub**, tape `NiBienvenu/modele_train`,
   choisis `entrainement_colab.ipynb`.
   (Dépôt privé : coche "Inclure les dépôts privés" et autorise GitHub.)
3. **Exécution → Modifier le type d'exécution → T4 GPU**.

### 3. Lancer l'entraînement, cellule par cellule

| Étape | À faire | Durée |
|---|---|---|
| 0 | Lancer les 4 cellules. La cellule 0.4 doit afficher **OK** partout | 2 min |
| 1, 2 | Rien : elles voient que les données et le tokenizer existent, et passent | quelques secondes |
| 3 | Phase A (petit modèle 14M) | 1 à 2 h |
| 4 | Agrandir : 14M -> 122M | 1 min |
| 5 | Phase B (grand modèle) | 6 à 10 h |
| 6 | Phase D (questions / réponses) | 1 à 2 h |
| 7 | Tests : notes par catégorie | 10 min |
| 8 | Essayer le modèle, télécharger le `.zip` | |

**Si Colab coupe** : relance les cellules de l'étape 0, puis la cellule de l'étape en cours.
L'entraînement **reprend tout seul** au dernier point sauvegardé dans Drive.

**Si "CUDA out of memory"** : dans `configs/phaseB.yaml` (ou `phaseD.yaml`), divise
`taille_lot` par 2 et multiplie `accumulation` par 2.

### 4. Lire les résultats

Tout est écrit dans `Mon Drive/modele/journaux/` :
- `phaseA.txt`, `phaseB.txt`, `phaseD.txt` : la **loss** (doit descendre) et des phrases d'exemple ;
- `notes_tests.txt` : la note de chaque catégorie de tests ;
- `resultats_tests.jsonl` : chaque question, la réponse du modèle et la bonne réponse.

Les catégories marquées **"NON (vraie preuve)"** n'ont jamais été vues à l'entraînement :
ce sont elles qui montrent si le modèle a vraiment **compris**.

---

## Organisation du projet

```
modele_train/
├── entrainement_colab.ipynb   le notebook qui enchaîne tout (Colab)
├── entrainement_local.ipynb   le même, sur ta machine (voir README_LOCAL.md)
├── requirements.txt
│
├── train.py                   entraînement (toutes les phases), reprise automatique
├── train_tokenizer.py         tokenizer français + kirundi (24 000 morceaux)
├── grow.py                    agrandit le modèle en gardant son savoir (testé : copie exacte)
├── evaluer.py                 fait passer les tests, notes exact + chrF, vitesse
├── construire_tests.py        fabrique tests/tests.jsonl
├── separation.py              réserve 10 % des phrases et 5 % des verbes pour les tests
├── configs/                   phaseA.yaml, phaseB.yaml, phaseD.yaml
│
├── gen_conjugaison_fr.py      français : 7 015 verbes, 12 temps
├── gen_conjugaison_rn.py      kirundi : 3 478 verbes, 14 temps
├── gen_accords_rn.py          kirundi : classes nominales, adjectifs, possessifs, démonstratifs
├── gen_syntaxe_rn.py          kirundi : pronoms objets, subordonnées, relatives (+ français accordé)
├── extraire_fsi.py            livre FSI (domaine public)
├── extraire_bible_db.py       Bible kirundi + français alignée (bases SQLite)
├── extraire_bible.py          ancienne version (PDF), remplacée par extraire_bible_db.py
├── extraire_dictionnaire_cox.py, fusionner_cox.py   dictionnaire de Cox (perfectifs, classes)
├── importer_dictionnaire.py   lexique français <-> kirundi, recherche des perfectifs
├── combiner_cours.py          phrases traduites des cours de Cox (Livres I et II)
│
├── data/                      listes vérifiées : verbes, noms, dictionnaire, cours
├── docs/grammaire/            règles de grammaire kirundi (étude des cours de Cox)
├── tests/                     le jeu de tests (568 questions)
└── A_METTRE_DANS_DRIVE/       (pas sur GitHub) données prêtes pour Google Drive
```

## Régénérer les données (seulement si on change un générateur)

Sur un ordinateur avec Python (ou dans Colab, étape 1 du notebook) :
```bash
pip install -r requirements.txt
python gen_conjugaison_rn.py --sortie A_METTRE_DANS_DRIVE/donnees/conjugaison_rn \
    --sft A_METTRE_DANS_DRIVE/donnees/qr/conjugaison_rn.jsonl
# ... voir l'étape 1 du notebook pour la liste complète
```
**Attention :** si les données changent beaucoup, il faut recréer le tokenizer **avant**
la phase A, jamais après.

## Droits

La Bible (versions kirundi et française), le dictionnaire et les cours d'Elizabeth Cox
sont protégés. Ils servent ici à un entraînement interne. Garde le dépôt **privé**, et
vérifie les droits avant de publier le modèle ou les données.
