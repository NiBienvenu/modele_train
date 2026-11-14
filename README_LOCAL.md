# Entraîner le modèle sur CETTE machine (GPU NVIDIA)

La méthode **Colab** reste décrite dans [README.md](README.md) et marche toujours.
Ce guide-ci fait la même chose **sur ton ordinateur**, avec sa carte graphique.

> Rien n'est partagé entre les deux méthodes à part le code : Colab écrit dans
> Google Drive (`/content/drive/MyDrive/...`, configs `configs/*.yaml`) ; la machine
> locale écrit dans `A_METTRE_DANS_DRIVE/` (configs `configs/local/*.yaml`).
> Les dossiers ont la **même forme** : tu peux commencer ici et finir sur Colab
> (ou l'inverse) en copiant le dossier `modele/`.

---

## 0. Ce que la machine a (et ce que ça change)

| | Cette machine | Colab (T4) |
|---|---|---|
| GPU | NVIDIA **T600 Laptop**, **4 Go**, 35 W | T4, 16 Go, 70 W |
| Calcul | float16 (pas de bfloat16 : génération Turing) | float16 |
| RAM / CPU | 31 Go / 20 cœurs | ~12 Go / 2 cœurs |
| Disque libre | ~230 Go | Drive : 15 Go gratuits |
| Coupures | aucune (si pas de veille) | toutes les quelques heures |

Ce qu'il faut retenir :

- **La mémoire (4 Go) est la vraie limite.** Les configs locales gardent le **même lot
  effectif** que Colab (donc les mêmes résultats), mais en plus petits morceaux :
  `taille_lot` divisé par 2 à 4, `accumulation` multipliée d'autant.
- **C'est environ 3 à 5 fois plus lent qu'un T4.** Ordre de grandeur :

  | Étape | Colab T4 | Cette machine (estimation) |
  |---|---|---|
  | Phase A (14M) | 1 à 2 h | 3 à 8 h |
  | Phase B (122M) | 6 à 10 h | **1 à 2 jours** |
  | Phase D | 1 à 2 h | 4 à 8 h |

  Mesure la vraie vitesse : la ligne `... | 2.31 s/pas` du journal, multipliée par
  le nombre de pas (`pas 50/12000`), donne le temps restant.

---

## 1. Installer (une seule fois) avec Anaconda

Anaconda est déjà installé (`~/anaconda3`). On crée un **environnement à part**,
`modele_train`, décrit dans `environment.yml` (Python 3.12 + PyTorch CUDA + `requirements.txt`).

```bash
cd ~/Documents/modele_train
conda env create -f environment.yml      # ~2,5 Go à télécharger, quelques minutes
conda activate modele_train
```

Règles avec conda :
- **Jamais dans `base`.** `base` sert à faire marcher conda lui-même ; si on y casse
  quelque chose, tout conda est cassé. Un environnement par projet.
- **PyTorch s'installe avec `pip`**, dans l'environnement conda (c'est ce que fait
  `environment.yml`). PyTorch ne publie plus de paquets sur les canaux conda, et
  `conda install pytorch` donne une version ancienne ou sans GPU.
- **Ne mélange pas** ensuite `conda install` et `pip install` pour les mêmes paquets :
  pour ajouter quelque chose, mets-le dans `requirements.txt` puis
  `conda env update -f environment.yml --prune`.
- Python **3.12** (et pas le 3.14 de `base`) : c'est la version de Colab, toutes les
  bibliothèques ont leurs paquets prêts pour elle.
- Tout casser et recommencer : `conda env remove -n modele_train`, puis `conda env create`.
- Pour générer les données (rarement) : `pip install -r requirements-donnees.txt` dans
  ce même environnement, puis `pip install --no-deps mlconjug3`.

Le script `entrainer_local.sh` active `modele_train` tout seul : pas besoin de
`conda activate` avant de le lancer.

Vérifier que PyTorch voit bien le GPU :

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# attendu : True NVIDIA T600 Laptop GPU
```

> Pas besoin d'installer `nvcc` ni le « CUDA Toolkit » : le paquet `torch` apporte
> tout ce qu'il faut. Seul le **pilote NVIDIA** (déjà installé) est nécessaire.

Les **données** et le **tokenizer** sont déjà dans `A_METTRE_DANS_DRIVE/`
(`donnees/`, `modele/tokenizer_fr_rn`) : pas besoin de les régénérer.

---

## 2. Préparer l'ordinateur (bonnes pratiques d'un portable)

À faire **avant chaque longue séance** :

1. **Brancher le chargeur.** Sur batterie, le GPU est bridé et la machine peut s'éteindre.
2. **Mode performance** (au lieu de « équilibré ») :
   ```bash
   powerprofilesctl set performance
   ```
   (Remettre `balanced` après.)
3. **Pas de veille** : le script `entrainer_local.sh` l'empêche tout seul (`systemd-inhibit`).
   Ne **ferme pas le capot** si ton système met en veille à la fermeture.
4. **La chaleur** : pose le portable sur une surface dure (pas un lit), aérations
   dégagées, idéalement surélevé. Surveille la température (voir §4) : au-dessus
   de ~85 °C, le GPU ralentit de lui-même.
5. **Libérer le GPU** : l'écran est géré par la puce Intel (mode `on-demand`), donc
   le GPU NVIDIA est presque libre. Ferme quand même les jeux, Blender, ou les
   navigateurs avec accélération lourde. Vérifie avec `nvidia-smi` : la mémoire
   utilisée doit être proche de 0 avant de lancer.
6. **Sauvegarde** : copie de temps en temps `A_METTRE_DANS_DRIVE/modele/` sur un
   disque externe ou sur Drive. Un disque qui lâche = des jours de calcul perdus.

---

## 3. Lancer l'entraînement

### Méthode A : avec Jupyter, comme sur Colab (conseillé pour suivre)

Le notebook `entrainement_local.ipynb` a les **mêmes étapes** que celui de Colab, plus une
cellule **0.4** qui dessine la courbe de la loss et l'état du GPU.

Une seule fois, déclarer l'environnement à Jupyter :

```bash
conda activate modele_train
python -m ipykernel install --user --name modele_train --display-name "Python (modele_train)"
```

Ensuite, à chaque fois :

```bash
cd ~/Documents/modele_train
conda activate modele_train
jupyter lab entrainement_local.ipynb
```

Le navigateur s'ouvre. En haut à droite, le noyau doit être **Python (modele_train)**.
Lance les cellules dans l'ordre, comme sur Colab.

Différences avec Colab à connaître :
- **Fermer l'onglet du navigateur n'arrête rien** : le noyau continue de tourner. Rouvre
  `http://localhost:8888` (ou relance `jupyter lab`) puis la cellule **0.4** pour revoir où
  on en est. L'affichage en direct de la cellule en cours, lui, est perdu (le journal non).
- **Fermer le terminal où tourne `jupyter lab`, redémarrer le noyau ou appuyer sur ■
  arrête l'entraînement.** Relance alors l'étape 0 puis la cellule de la phase : elle reprend
  au dernier point sauvegardé. Pour ne pas dépendre du terminal :
  `nohup jupyter lab --no-browser > jupyter.log 2>&1 &` puis ouvre l'adresse affichée dans
  `jupyter.log`.
- Pendant qu'une cellule tourne, on ne peut pas en lancer une autre dans le même notebook :
  pour voir la courbe en même temps, utilise un **terminal** (`tail -f`, voir §4) ou ouvre
  un 2e notebook (Fichier → Nouveau → Notebook) avec le même code que la cellule 0.4.

### Méthode B : avec le script (pour lancer et oublier)

Tout d'un coup (phase A → agrandissement → phase B → phase D → tests) :

```bash
cd ~/Documents/modele_train
nohup ./entrainer_local.sh > entrainement.log 2>&1 &
```

`nohup ... &` : l'entraînement continue même si tu fermes le terminal.

Ou une étape à la fois :

```bash
./entrainer_local.sh phaseA
./entrainer_local.sh grow
./entrainer_local.sh phaseB
./entrainer_local.sh phaseD
./entrainer_local.sh tests
```

Ce que fait le script, comme le notebook :
- il **saute** les étapes déjà finies ;
- si une phase a été **interrompue** (Ctrl+C, coupure de courant, redémarrage), relance
  la même commande : elle **reprend** au dernier point sauvegardé (`derniere/`, tous les
  1000 pas) ;
- il écrit les journaux dans `A_METTRE_DANS_DRIVE/modele/journaux/`.

**Arrêter proprement** : `pkill -f train.py` (ou Ctrl+C si lancé sans `nohup`).
Tu perds au plus le travail depuis la dernière sauvegarde.

---

## 4. Surveiller

Dans un autre terminal :

```bash
tail -f A_METTRE_DANS_DRIVE/modele/journaux/phaseB.txt     # la loss, les phrases d'exemple
watch -n 5 nvidia-smi                                      # mémoire, température, % d'usage
```

Signes que tout va bien :
- **GPU-Util** proche de 90-100 % (sinon, le GPU attend les données ou chauffe trop) ;
- **mémoire** sous ~3,8 Go / 4 Go ;
- **température** sous ~85 °C ;
- la **loss** descend, et les phrases d'exemple deviennent du vrai français / kirundi.

---

## 5. Si ça ne marche pas

| Problème | Solution |
|---|---|
| `CUDA out of memory` | Dans `configs/local/phaseX.yaml` : divise `taille_lot` par 2, multiplie `accumulation` par 2 (ex. phase B : `2` et `64`). Le résultat reste le même, c'est juste plus lent. |
| Toujours trop de mémoire avec `taille_lot: 1` | Baisse `longueur` à 256 (phases B et D). |
| `torch.cuda.is_available()` donne `False` | `nvidia-smi` doit marcher ; sinon réinstalle le pilote. Vérifie aussi que tu es dans le bon environnement (`conda activate modele_train` ; `which python` doit finir par `envs/modele_train/bin/python`). |
| Très lent, GPU-Util bas | Chargeur branché ? Mode performance ? Température ? |
| Une phase B de 2 jours, c'est trop | Voir §6. |

---

## 6. Stratégies pour gagner du temps

1. **Mixte (conseillé)** : phase A ici (petit modèle, rapide même sur 4 Go), puis copie
   `A_METTRE_DANS_DRIVE/modele/phaseA/` dans `Mon Drive/modele/phaseA/` et fais la
   phase B sur Colab. Le notebook voit que la phase A est finie et passe à la suite.
2. **Un modèle un peu plus petit en local** : à l'agrandissement, au lieu de 122M :
   ```bash
   HIDDEN=512 LAYERS=8 HEADS=8 ./entrainer_local.sh grow
   ```
   Environ 3 fois moins de calcul. Moins fort, mais suffisant pour tester des idées.
3. **Essais rapides** avant une longue séance : mets `epoques: 1` et vérifie que la loss
   descend sur les premières centaines de pas. Ne lance la séance complète qu'après.
4. **Ne pas utiliser le GPU pendant ce temps** (jeu, vidéo…) : chaque Mo compte.

---

## 7. Résultats et utilisation du modèle

Tout est dans `A_METTRE_DANS_DRIVE/modele/` :

```
modele/
├── tokenizer_fr_rn/      (ne change jamais)
├── phaseA/  finale/ meilleure/ derniere/
├── grand_init/           (le modèle agrandi, avant la phase B)
├── phaseB/  finale/ meilleure/ derniere/
├── phaseD/  finale/ meilleure/ derniere/    <- le modèle qui répond aux questions
└── journaux/ phaseA.txt phaseB.txt phaseD.txt notes_tests.txt resultats_tests.jsonl
```

- `finale/` : à la fin de la phase ; `meilleure/` : la meilleure loss d'évaluation ;
  `derniere/` : le point de reprise (contient aussi l'état de l'optimiseur, plus gros).
- Ces dossiers font plusieurs Go : ils sont dans `.gitignore`, ne les mets **pas** dans git.

Essayer le modèle (après la phase D) :

```bash
python evaluer.py --modele A_METTRE_DANS_DRIVE/modele/phaseD/finale \
    --categorie traduction_rn_fr --montrer 15
```

Les règles de droits du [README.md](README.md#droits) s'appliquent aussi ici.
