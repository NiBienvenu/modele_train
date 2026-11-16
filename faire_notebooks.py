#!/usr/bin/env python
"""
faire_notebooks.py : écrit les 3 notebooks du modèle zéro (mêmes étapes, préparation différente).

    python3 faire_notebooks.py

    zero_colab.ipynb    Google Colab  (dataset dans Mon Drive)
    zero_kaggle.ipynb   Kaggle        (dataset ajouté comme "Input")
    zero_machine.ipynb  cette machine (dataset dans A_METTRE_DANS_DRIVE/)

Une copie est aussi mise dans A_METTRE_DANS_DRIVE/dataset/notebooks/.
Pour changer une étape, modifie ce fichier puis relance-le (ne modifie pas les .ipynb à la main).
"""

import json
from pathlib import Path

ICI = Path(__file__).parent


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(True)}


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": t.strip("\n").splitlines(True)}


# ---------------------------------------------------------------------------
# Étape 0 : différente pour chaque plateforme. Elle définit DATASET, TRAVAIL, CODE.
# ---------------------------------------------------------------------------
INTRO = """
# Modèle ZÉRO : français + kirundi ({plateforme})

Le modèle part de **rien** : poids au hasard, nouveau tokenizer. Tout vient du dossier **`dataset`**
(textes, dialogues, tests et code), le même pour Colab, Kaggle et la machine.

| Étape | Ce qui se passe | Durée ({materiel}) |
|---|---|---|
| 0 | Préparer : GPU, dossiers | {d0} |
| 1 | Télécharger les données publiques (Wikipédia FR, OpenAssistant, PIAF, Aya) | {d1} |
| 2 | Créer le tokenizer (32 000 morceaux, moitié kirundi, moitié français) | {d2} |
| 3 | **Lecture** : le modèle lit tous les textes | {d3} |
| 4 | **Discussion** : il apprend à répondre, traduire, lire un document | {d4} |
| 5 | Tests sur des exemples jamais vus | {d5} |
| 6 | Essayer le modèle et récupérer le `.zip` pour le backend | |

Ces durées sont des **estimations**. La vraie vitesse s'affiche pendant l'entraînement (`s/pas` dans le journal).
{avant}
"""

# Durées estimées par plateforme. La lecture fait ~1 500 pas de 131 000 morceaux (~200 millions de morceaux).
DUREES = {
    "colab": dict(materiel="GPU T4 gratuit", d0="2 min", d1="30 à 60 min", d2="15 min",
                  d3="12 à 15 h, en 3 ou 4 sessions (Colab coupe après quelques heures)", d4="2 à 3 h",
                  d5="15 min"),
    "kaggle": dict(materiel="GPU T4 ou P100, 30 h par semaine", d0="2 min", d1="20 à 40 min (connexion rapide)",
                   d2="10 min", d3="12 à 15 h, donc 2 lancements (maximum 12 h par lancement)",
                   d4="2 à 3 h", d5="15 min"),
    "machine": dict(materiel="T600 4 Go, environ 5 fois plus lent qu'un T4", d0="1 min",
                    d1="selon ta connexion internet", d2="5 à 10 min (20 cœurs)",
                    d3="3 à 4 jours sans arrêt", d4="10 à 15 h", d5="1 h"),
}

COLAB = [
    md(INTRO.format(plateforme="Google Colab", **DUREES["colab"], avant="""
**Avant de commencer** : glisse le dossier `A_METTRE_DANS_DRIVE/dataset` dans **Mon Drive** (à la racine).
Menu **Exécution > Modifier le type d'exécution > GPU T4**.

**Si Colab coupe** : relance l'étape 0, puis la cellule de l'étape en cours. Tout reprend au dernier point sauvegardé dans Drive.
""")),
    md("## Étape 0 : préparer Colab"),
    code("""
!nvidia-smi --query-gpu=name,memory.total --format=csv
from google.colab import drive
drive.mount('/content/drive')

import os, shutil
DATASET = "/content/drive/MyDrive/dataset"        # lecture
TRAVAIL = "/content/drive/MyDrive/modele_zero"    # écriture (survit aux coupures)
# Le code est copié hors de Drive (plus rapide, et Drive reste propre)
CODE = "/content/code"
shutil.copytree(f"{DATASET}/code", CODE, dirs_exist_ok=True)
!pip install -q -r {CODE}/requirements.txt
"""),
]

KAGGLE = [
    md(INTRO.format(plateforme="Kaggle", **DUREES["kaggle"], avant="""
**Avant de commencer (une seule fois)** :
1. Sur kaggle.com : **Datasets > New Dataset**, glisse le dossier `A_METTRE_DANS_DRIVE/dataset`, garde-le **privé**.
2. Dans ce notebook : **Add Input** > ton dataset.
3. **Settings** : Accelerator = **GPU T4 x2** (ou P100), **Internet = On** (il faut un téléphone vérifié).

**Kaggle arrête une session après 12 h**, et la lecture dure 12 à 15 h. Fais comme ça :
1. **Save Version > Save & Run All (Commit)** : le notebook tourne seul, même navigateur fermé.
2. Quand il s'arrête, ses fichiers (`/kaggle/working/modele_zero`) sont gardés dans l'onglet **Output**.
3. Pour continuer : **Add Input** > *Notebook Output* > ce notebook, puis **Save & Run All** encore une fois.
   L'étape 0 recopie le travail précédent, et l'entraînement reprend où il s'était arrêté.
""")),
    md("## Étape 0 : préparer Kaggle"),
    code("""
!nvidia-smi --query-gpu=name,memory.total --format=csv
import glob, os, shutil

# Le dataset : on cherche code/train.py dans les "Inputs" (le nom exact du dossier peut changer)
trouves = glob.glob("/kaggle/input/**/code/train.py", recursive=True)
if not trouves:
    raise SystemExit("Dataset introuvable : clique sur 'Add Input' et ajoute ton dataset.")
DATASET = os.path.dirname(os.path.dirname(trouves[0]))
TRAVAIL = "/kaggle/working/modele_zero"           # écriture (gardé dans 'Output' à la fin)
CODE = "/kaggle/working/code"
shutil.copytree(f"{DATASET}/code", CODE, dirs_exist_ok=True)

# Reprise : si la sortie d'une version précédente est ajoutée en Input, on la recopie
if not os.path.exists(TRAVAIL):
    anciens = sorted(glob.glob("/kaggle/input/**/modele_zero", recursive=True))
    if anciens:
        print("Reprise depuis", anciens[0], "(copie, 1 à 3 min)")
        shutil.copytree(anciens[0], TRAVAIL)
print("DATASET =", DATASET)
"""),
]

MACHINE = [
    md(INTRO.format(plateforme="sur cette machine", **DUREES["machine"], avant="""
**Avant de commencer** : ouvre ce notebook depuis le dossier `modele_train/`, dans l'environnement
qui a PyTorch avec CUDA (voir `README_LOCAL.md`). Le dataset est déjà là : `A_METTRE_DANS_DRIVE/dataset`.

**Si ça s'arrête** : relance l'étape 0, puis la cellule de l'étape en cours. Tout reprend au dernier point sauvegardé.
""")),
    md("## Étape 0 : préparer la machine"),
    code("""
!nvidia-smi --query-gpu=name,memory.total --format=csv
import os
ICI = os.path.abspath(".")
if not os.path.exists(f"{ICI}/A_METTRE_DANS_DRIVE/dataset/code/train.py"):
    raise SystemExit("Lance ce notebook depuis le dossier modele_train/ (et lance d'abord preparer_donnees_zero.py).")
DATASET = f"{ICI}/A_METTRE_DANS_DRIVE/dataset"
TRAVAIL = f"{ICI}/A_METTRE_DANS_DRIVE/modele_zero"
CODE = f"{TRAVAIL}/code"   # une COPIE du code : on peut l'adapter sans toucher au dataset
import re, shutil, torch
shutil.copytree(f"{DATASET}/code", CODE, dirs_exist_ok=True)

# Petite carte graphique (ex : T600, 4 Go) : mêmes calculs, mais par plus petits morceaux
if not torch.cuda.is_available():
    raise SystemExit("PyTorch ne voit pas de GPU NVIDIA : vérifie l'environnement (README_LOCAL.md).")
go = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU : {torch.cuda.get_device_name(0)}, {go:.0f} Go")
if go < 10:
    for nom, lot, acc in [("zero_pretrain", 1, 128), ("zero_chat", 1, 32)]:
        chemin = f"{CODE}/configs/{nom}.yaml"
        texte = open(chemin, encoding="utf-8").read()
        texte = re.sub(r"(?m)^taille_lot: .*", f"taille_lot: {lot}", texte)
        texte = re.sub(r"(?m)^accumulation: .*", f"accumulation: {acc}   # adapté à {go:.0f} Go", texte)
        if "gradient_checkpointing" not in texte:
            texte += "\\ngradient_checkpointing: true   # économise la mémoire (un peu plus lent)\\n"
        open(chemin, "w", encoding="utf-8").write(texte)
    print("Configs adaptées à une petite carte. ATTENTION : la lecture sera environ 5 fois plus longue",
          "qu'un T4 (plusieurs jours). Colab ou Kaggle sont plus rapides pour l'étape 3.")
"""),
]

# ---------------------------------------------------------------------------
# Étapes 1 à 6 : identiques partout
# ---------------------------------------------------------------------------
COMMUN = [
    code("""
# 0.b  Commun à toutes les plateformes : variables lues par train.py ({DATASET} et {TRAVAIL} dans les configs)
os.environ["DATASET"], os.environ["TRAVAIL"] = DATASET, TRAVAIL
os.makedirs(f"{TRAVAIL}/journaux", exist_ok=True)
os.chdir(CODE)
J = f"{TRAVAIL}/journaux"

def fait(chemin):
    return os.path.exists(chemin)

def verifier(chemin, etape):
    if os.path.exists(chemin):
        print(f"OK : {etape} terminée -> {chemin}")
        return
    raise SystemExit(f"ÉCHEC : {etape} n'a pas produit {chemin}. Regarde {J}/{etape}.txt. "
                     "Ne passe PAS à l'étape suivante.")

def meilleur_modele():
    for ph in ("chat", "lecture"):
        for sous in ("finale", "meilleure", "derniere"):
            if os.path.exists(f"{TRAVAIL}/{ph}/{sous}/config.json"):
                return f"{TRAVAIL}/{ph}/{sous}"
    raise SystemExit("Aucun modèle trouvé : lance d'abord l'étape 3.")

for d in ("lecture_rn", "lecture_fr", "lecture_bilingue", "grammaire", "dialogues"):
    print(("OK      " if fait(f"{DATASET}/{d}") else "MANQUANT"), f"{DATASET}/{d}")
print()
for nom, chemin in [("1. Données publiques", f"{TRAVAIL}/publics/lecture_fr/FINI"),
                    ("2. Tokenizer", f"{TRAVAIL}/tokenizer/tokenizer.json"),
                    ("3. Lecture", f"{TRAVAIL}/lecture/finale"), ("4. Discussion", f"{TRAVAIL}/chat/finale")]:
    reprise = fait(chemin.replace("finale", "derniere"))
    print(f"{nom:22s}: " + ("FINI" if fait(chemin) else "en cours (reprise possible)" if reprise else "pas commencé"))
"""),
    md("## Étape 1 : données publiques\nUne source qui ne se télécharge pas est sautée, les autres continuent."),
    code("""
os.chdir(CODE)
if not fait(f"{TRAVAIL}/publics/lecture_fr/FINI"):
    !python -u collecte/telecharger_publics.py --sortie {TRAVAIL}/publics --mots 150000000 2>&1 | grep -v "examples/s" | tee {J}/publics.txt
!ls {TRAVAIL}/publics/lecture_fr | head -3; wc -l {TRAVAIL}/publics/dialogues/*.jsonl
"""),
    md("""
## Étape 2 : le tokenizer
Le kirundi est lu 10 fois et un seul fichier Wikipédia une fois : les deux langues ont à peu près **la même place**.
Regarde la vérification à la fin : les phrases kirundi doivent être coupées en morceaux qui ont du sens.
"""),
    code("""
os.chdir(CODE)
if not fait(f"{TRAVAIL}/tokenizer/tokenizer.json"):
    !python train_tokenizer.py --taille_vocab 32000 \\
        --source "{DATASET}/lecture_rn/*.txt=10" --source "{DATASET}/lecture_bilingue/*.txt=3" \\
        --source "{TRAVAIL}/publics/lecture_fr/wikipedia_fr_001.txt=1" --source "{DATASET}/lecture_fr/*.txt=1" \\
        --sortie {TRAVAIL}/tokenizer 2>&1 | tee {J}/tokenizer.txt
verifier(f"{TRAVAIL}/tokenizer/tokenizer.json", "tokenizer")
"""),
    md("""
## Étape 3 : lecture (la plus longue)
**Si ça coupe** : relance l'étape 0, puis cette cellule. Ça reprend tout seul.
Si *CUDA out of memory* : dans `code/configs/zero_pretrain.yaml`, mets `taille_lot: 4` et `accumulation: 32`.
"""),
    code("""
os.chdir(CODE)
if not fait(f"{TRAVAIL}/lecture/finale"):
    !python -u train.py --config configs/zero_pretrain.yaml 2>&1 | grep --line-buffered -vE 'it/s\\]|examples/s' | tee -a {J}/lecture.txt
verifier(f"{TRAVAIL}/lecture/finale", "lecture")
"""),
    md("## Étape 4 : discussion (répondre, traduire, lire un document)"),
    code("""
os.chdir(CODE)
if not fait(f"{TRAVAIL}/chat/finale"):
    !python -u train.py --config configs/zero_chat.yaml 2>&1 | grep --line-buffered -vE 'it/s\\]|examples/s' | tee -a {J}/chat.txt
verifier(f"{TRAVAIL}/chat/finale", "chat")
"""),
    md("""
## Étape 5 : tests
**exact** = réponse parfaite, **chrF** = ressemblance (0 à 100, la mesure habituelle des traductions).
Seules les catégories **« NON (vraie preuve) »** montrent si le modèle a compris : il ne les a jamais vues.
La plus importante : **traductions_bible** (versets jamais vus, dans les deux sens).
"""),
    code("""
os.chdir(CODE)
!python -u evaluer.py --modele {meilleur_modele()} --tests tests/dialogues_test.jsonl --max 100 --montrer 3 \\
    --sortie {J}/resultats_tests.jsonl 2>&1 | tee {J}/notes_tests.txt
"""),
    md("## Étape 6 : essayer le modèle"),
    code("""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
appareil = "cuda" if torch.cuda.is_available() else "cpu"
dossier = meilleur_modele()
tok = AutoTokenizer.from_pretrained(dossier)
model = AutoModelForCausalLM.from_pretrained(dossier).to(appareil).eval()
historique = ""

def demander(question, garder=True):
    \"\"\"Pose une question. garder=True : le modèle se souvient des tours d'avant.\"\"\"
    global historique
    prompt = f"{historique}<|user|> {question} <|assistant|>"
    ids = tok(prompt, return_tensors="pt", return_token_type_ids=False).to(appareil)
    out = model.generate(**ids, max_new_tokens=150, do_sample=True, temperature=0.3, top_p=0.9,
                         eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    reponse = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    if garder:
        historique = f"{prompt} {reponse}<|endoftext|>"
    return reponse

print(dossier)
for q in ["Bonjour", "Bwakeye.", "N'amaki?", "Comment dit-on merci en kirundi ?", "Qui es-tu ?",
          "Traduis en français : Umwana n'umukobwa."]:
    print(q, "->", demander(q, garder=False))
"""),
    code("""
# Une vraie conversation (historique = "" pour recommencer)
historique = ""
print(demander("Bonjour"))
print(demander("Comment dit-on au revoir en kirundi ?"))
print(demander("Merci"))
"""),
    md("## Étape 7 : le `.zip` pour le backend\nÀ mettre dans `litgpt/models/local/`, puis `python scripts/convert_local_model.py <dossier>`."),
]

FIN = {
    "colab": code("""
dossier_modele = meilleur_modele()
!cd "{dossier_modele}" && zip -qr /content/modele_zero_fr_rn.zip . -x "etat_entrainement.pt" && ls -lh /content/modele_zero_fr_rn.zip
!cp /content/modele_zero_fr_rn.zip {TRAVAIL}/
from google.colab import files
files.download("/content/modele_zero_fr_rn.zip")
"""),
    "kaggle": code("""
dossier_modele = meilleur_modele()
!cd "{dossier_modele}" && zip -qr /kaggle/working/modele_zero_fr_rn.zip . -x "etat_entrainement.pt"
!ls -lh /kaggle/working/modele_zero_fr_rn.zip
print("Après 'Save Version', le .zip est dans l'onglet Output de ce notebook : clique dessus pour le télécharger.")
"""),
    "machine": code("""
dossier_modele = meilleur_modele()
!cd "{dossier_modele}" && zip -qr {TRAVAIL}/modele_zero_fr_rn.zip . -x "etat_entrainement.pt" && ls -lh {TRAVAIL}/modele_zero_fr_rn.zip
"""),
}


def ecrire(nom, debut, plateforme):
    meta = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}}
    if plateforme == "colab":
        meta["accelerator"] = "GPU"
        meta["colab"] = {"gpuType": "T4", "provenance": []}
    nb = {"cells": debut + COMMUN + [FIN[plateforme]], "metadata": meta, "nbformat": 4, "nbformat_minor": 5}
    for dossier in (ICI, ICI / "A_METTRE_DANS_DRIVE/dataset/notebooks"):
        dossier.mkdir(parents=True, exist_ok=True)
        (dossier / nom).write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Écrit :", nom)


if __name__ == "__main__":
    ecrire("zero_colab.ipynb", COLAB, "colab")
    ecrire("zero_kaggle.ipynb", KAGGLE, "kaggle")
    ecrire("zero_machine.ipynb", MACHINE, "machine")
