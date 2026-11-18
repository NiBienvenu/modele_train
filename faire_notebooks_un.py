#!/usr/bin/env python
"""
faire_notebooks_un.py : écrit les 3 notebooks du MODÈLE UN (suite du modèle zéro).

    python3 faire_notebooks_un.py        (appelé aussi par preparer_donnees_un.py)

    un_kaggle.ipynb   Kaggle  (3 Inputs : dataset, dataset-un, sortie du notebook zero_kaggle)
    un_colab.ipynb    Colab   (Mon Drive : dataset/, dataset_un/, modele_zero/)
    un_machine.ipynb  cette machine

Une copie va dans datasets/dataset_un/notebooks/. Ne pas modifier les .ipynb à la main.
"""

import json
from pathlib import Path

ICI = Path(__file__).parent


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(True)}


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": t.strip("\n").splitlines(True)}


INTRO = """
# Modèle UN : français + kirundi ({plateforme})

On **continue le modèle zéro** (on ne repart pas de rien) : même tokenizer, mêmes connaissances,
puis il relit l'ancien et lit le nouveau (`dataset_un` : textes kirundi vérifiés, insiguro).

| Étape | Ce qui se passe |
|---|---|
| 0 | Préparer : trouver les dossiers, l'heure de fin de session |
| 1 | Données publiques (Wikipédia FR...) : reprises du zéro, ou retéléchargées |
| 2 | **Relecture** : ancien (moins de Bible) + nouveau |
| 3 | **Discussion** : répondre, traduire, insiguro |
| 4 | Tests (les mêmes que le zéro + FLORES) |
| 5 | Essayer le modèle, puis le `.zip` |

**Sauvegardes** : petite sauvegarde tous les 50 pas, **copie complète toutes les 4 h**
(`sauvegarde_4h`, `sauvegarde_8h`), et **pause propre avant la fin de la session**.
Si une étape n'est pas finie à la pause, la **version suivante** du notebook reprend au même pas.
{avant}
"""

KAGGLE = [
    md(INTRO.format(plateforme="Kaggle", avant="""
**Avant de lancer (Inputs)** : **Add Input** 2 fois :
1. Datasets > **modele-train** (dernière version). À sa racine : `dataset/` (modèle zéro) et `dataset_un/` (modèle un).
2. Notebooks > **zero_kaggle** (sa sortie contient `modele_zero/lecture/finale`, le point de départ).

**Settings** : Accelerator = **GPU T4 x2**, Internet = **On**. Puis **Save Version > Save & Run All**.

**Session de 12 h** : le notebook s'arrête de lui-même vers 11 h (pause propre), les fichiers restent
dans l'onglet **Output**. **Pour continuer** : Add Input > Notebooks > **ce notebook (un_kaggle)**,
puis Save & Run All encore une fois : il recopie `modele_un` et reprend.
""")),
    md("## Étape 0 : préparer Kaggle"),
    code("""
import glob, os, shutil, time
DEBUT = time.time()
HEURES_SESSION = 11.25     # Kaggle coupe à 12 h : on garde du temps pour les tests et la sortie
!nvidia-smi --query-gpu=name,memory.total --format=csv

# Ce que Kaggle a mis dans les Inputs (suit aussi les liens, Kaggle en utilise)
TOUS = []
for racine, dossiers, fichiers in os.walk("/kaggle/input", followlinks=True):
    niveau = racine[len("/kaggle/input"):].count("/")
    if niveau <= 5:
        print("  " * niveau + os.path.basename(racine) + "/", f"({len(fichiers)} fichiers)" if fichiers else "")
    TOUS += [os.path.join(racine, f) for f in fichiers]
print(len(TOUS), "fichiers en tout dans /kaggle/input")

def trouver(fin, quoi, sauf=None):
    # cherche un fichier dont le chemin se termine par `fin` (peu importe la profondeur)
    t = sorted(x for x in TOUS if x.endswith("/" + fin) and not (sauf and sauf in x))
    if not t:
        raise FileNotFoundError(f"\\n\\nIntrouvable : {quoi} (fichier .../{fin}).\\n"
                                f"Clique 'Add Input' et ajoute-le, puis relance. Liste des Inputs ci-dessus.")
    return t[0]

DATASET = os.path.dirname(os.path.dirname(trouver("dialogues/salutations_rn.jsonl", "ancien dataset")))
DATASET_UN = os.path.dirname(os.path.dirname(trouver("insiguro/insiguro.txt", "dataset-un")))
# Le modèle zéro : sortie du notebook zero_kaggle (avec ou sans le dossier modele_zero/ autour)
ZERO = os.path.dirname(os.path.dirname(os.path.dirname(trouver(
    "lecture/finale/config.json",
    "le modèle zéro (Add Input > Your Work > Notebooks > zero_kaggle)", sauf="modele_un"))))
TRAVAIL = "/kaggle/working/modele_un"
CODE = "/kaggle/working/code_un"
shutil.copytree(f"{DATASET_UN}/code", CODE, dirs_exist_ok=True)

# Reprise : sortie d'une version précédente de CE notebook ajoutée en Input
if not os.path.exists(TRAVAIL):
    anciens = sorted(glob.glob("/kaggle/input/**/modele_un/lecture", recursive=True))
    if anciens:
        print("Reprise depuis", os.path.dirname(anciens[0]), "(copie, quelques minutes)")
        shutil.copytree(os.path.dirname(anciens[0]), TRAVAIL)
PUBLICS = f"{ZERO}/publics" if glob.glob(f"{ZERO}/publics/lecture_fr/wikipedia_fr_001.txt") else f"{TRAVAIL}/publics"
"""),
]

COLAB = [
    md(INTRO.format(plateforme="Google Colab", avant="""
**Avant de lancer** : dans **Mon Drive** (à la racine) il faut :
1. `dataset/` (l'ancien, déjà là si tu as utilisé Colab pour le zéro)
2. `dataset_un/` (le dossier `datasets/dataset_un` de la machine)
3. `modele_zero/lecture/finale/` : le modèle zéro **avant la discussion**. Sur Kaggle : onglet Output du
   notebook zero_kaggle > `modele_zero/lecture/finale` > télécharger, puis le mettre dans Drive.

Menu **Exécution > Modifier le type d'exécution > GPU T4**, puis **Exécuter tout**.
**Si Colab coupe** : relance tout (Exécuter tout). Chaque étape reprend au dernier point sauvegardé dans Drive.
""")),
    md("## Étape 0 : préparer Colab"),
    code("""
import glob, os, shutil, time
DEBUT = time.time()
HEURES_SESSION = 0         # 0 = pas de limite connue : Colab peut couper, on reprend grâce aux sauvegardes
!nvidia-smi --query-gpu=name,memory.total --format=csv
from google.colab import drive
drive.mount('/content/drive')
DATASET = "/content/drive/MyDrive/dataset"
DATASET_UN = "/content/drive/MyDrive/dataset_un"
ZERO = "/content/drive/MyDrive/modele_zero"
TRAVAIL = "/content/drive/MyDrive/modele_un"      # écriture dans Drive (survit aux coupures)
for d, quoi in [(f"{DATASET}/dialogues", "dataset"), (f"{DATASET_UN}/insiguro", "dataset_un"),
                (f"{ZERO}/lecture/finale/config.json", "modele_zero/lecture/finale")]:
    if not os.path.exists(d):
        raise SystemExit(f"Manque dans Mon Drive : {quoi} (voir en haut).")
CODE = "/content/code_un"
shutil.copytree(f"{DATASET_UN}/code", CODE, dirs_exist_ok=True)
!pip install -q -r {CODE}/requirements.txt
PUBLICS = f"{ZERO}/publics" if glob.glob(f"{ZERO}/publics/lecture_fr/wikipedia_fr_001.txt") else f"{TRAVAIL}/publics"
"""),
]

MACHINE = [
    md(INTRO.format(plateforme="sur cette machine", avant="""
**Avant de lancer** : ouvre ce notebook depuis `modele_train/`, dans l'environnement PyTorch + CUDA
(`README_LOCAL.md`). Il faut `datasets/dataset`, `datasets/dataset_un` et le modèle zéro
dans `modeles/modele_zero/lecture/finale` (téléchargé depuis l'onglet Output de zero_kaggle).
Une carte de moins de 10 Go marche, mais lentement (plusieurs jours).
""")),
    md("## Étape 0 : préparer la machine"),
    code("""
import glob, os, re, shutil, time, torch
DEBUT = time.time()
HEURES_SESSION = 0
!nvidia-smi --query-gpu=name,memory.total --format=csv
ICI = os.path.abspath(".")
DATASET = f"{ICI}/datasets/dataset"
DATASET_UN = f"{ICI}/datasets/dataset_un"
ZERO = f"{ICI}/modeles/modele_zero"
TRAVAIL = f"{ICI}/modeles/modele_un"
for d, quoi in [(f"{DATASET}/dialogues", "dataset"), (f"{DATASET_UN}/insiguro", "dataset_un (preparer_donnees_un.py)"),
                (f"{ZERO}/lecture/finale/config.json", "modele_zero/lecture/finale")]:
    if not os.path.exists(d):
        raise SystemExit(f"Manque : {quoi}.")
CODE = f"{TRAVAIL}/code"
shutil.copytree(f"{DATASET_UN}/code", CODE, dirs_exist_ok=True)
if not torch.cuda.is_available():
    raise SystemExit("PyTorch ne voit pas de GPU NVIDIA (README_LOCAL.md).")
go = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU : {torch.cuda.get_device_name(0)}, {go:.0f} Go")
if go < 10:   # petite carte : mêmes calculs, par plus petits morceaux
    for nom, lot, acc in [("un_lecture", 1, 128), ("un_chat", 1, 32)]:
        chemin = f"{CODE}/configs/{nom}.yaml"
        texte = open(chemin, encoding="utf-8").read()
        texte = re.sub(r"(?m)^taille_lot: .*", f"taille_lot: {lot}", texte)
        texte = re.sub(r"(?m)^accumulation: .*", f"accumulation: {acc}", texte)
        texte += "\\ngradient_checkpointing: true\\n"
        open(chemin, "w", encoding="utf-8").write(texte)
    print("Configs adaptées à une petite carte (beaucoup plus lent qu'un T4).")
PUBLICS = f"{ZERO}/publics" if glob.glob(f"{ZERO}/publics/lecture_fr/wikipedia_fr_001.txt") else f"{TRAVAIL}/publics"
"""),
]

COMMUN = [
    code("""
# 0.b  Commun : variables lues par train.py, heure de fin, état de chaque étape
import torch
os.makedirs(f"{TRAVAIL}/journaux", exist_ok=True)
J = f"{TRAVAIL}/journaux"
os.environ.update(DATASET=DATASET, DATASET_UN=DATASET_UN, ZERO=ZERO, TRAVAIL=TRAVAIL, PUBLICS=PUBLICS)
if HEURES_SESSION:
    os.environ["FIN_SESSION"] = str(DEBUT + HEURES_SESSION * 3600)
    print("Pause prévue vers", time.strftime("%H:%M", time.localtime(DEBUT + HEURES_SESSION * 3600 - 25 * 60)))
NB_GPU = max(1, torch.cuda.device_count())
LANCER = f"torchrun --nproc_per_node {NB_GPU} train.py" if NB_GPU > 1 else "python -u train.py"
print(f"{NB_GPU} GPU -> {LANCER}")
os.chdir(CODE)
PAUSE = False

def fait(chemin):
    return os.path.exists(chemin)

def apres(etape, dossier):
    \"\"\"Après une étape : finie, en pause (reprise à la version suivante), ou en échec.\"\"\"
    global PAUSE
    if fait(f"{dossier}/finale/config.json"):
        print(f"OK : {etape} terminée -> {dossier}/finale")
    elif fait(f"{dossier}/PAUSE"):
        PAUSE = True
        print(f"PAUSE : {etape} pas finie (temps de session). Les étapes suivantes attendent la version suivante.")
    else:
        raise SystemExit(f"ÉCHEC : {etape}. Regarde {J}/{etape}.txt")

def meilleur_modele():
    for ph in ("chat", "lecture"):
        for sous in ("finale", "meilleure"):
            if fait(f"{TRAVAIL}/{ph}/{sous}/config.json"):
                return f"{TRAVAIL}/{ph}/{sous}"
    return f"{ZERO}/lecture/finale"

print("DATASET    =", DATASET); print("DATASET_UN =", DATASET_UN); print("ZERO       =", ZERO)
print("PUBLICS    =", PUBLICS); print("TRAVAIL    =", TRAVAIL)
for nom, ph in [("Relecture", "lecture"), ("Discussion", "chat")]:
    d = f"{TRAVAIL}/{ph}"
    print(f"{nom:11s}: " + ("FINI" if fait(f"{d}/finale") else "en cours (reprise)" if fait(f"{d}/derniere")
                             else "pas commencé"))
"""),
    md("## Étape 1 : données publiques\nReprises de la sortie du zéro si elles y sont, sinon téléchargées (1 fichier Wikipédia suffit)."),
    code("""
os.chdir(CODE)
if not glob.glob(f"{PUBLICS}/lecture_fr/wikipedia_fr_001.txt"):
    !python -u collecte/telecharger_publics.py --sortie {PUBLICS} --mots 20000000 2>&1 | grep -v "examples/s" | tee {J}/publics.txt
!ls {PUBLICS}/lecture_fr | head -3; ls {PUBLICS}/dialogues
"""),
    md("""
## Étape 2 : relecture (continue le modèle zéro)
Regarde les phrases test : moins de répétitions et moins de style biblique qu'avec le zéro.
La dernière (« Guhodoka : ») montre s'il retient les insiguro.
"""),
    code("""
os.chdir(CODE)
if not PAUSE and not fait(f"{TRAVAIL}/lecture/finale"):
    !{LANCER} --config configs/un_lecture.yaml 2>&1 | grep --line-buffered -vE 'it/s\\]|examples/s' | tee -a {J}/lecture.txt
apres("lecture", f"{TRAVAIL}/lecture")
"""),
    md("## Étape 3 : discussion (répondre, traduire, insiguro)"),
    code("""
os.chdir(CODE)
if not PAUSE and not fait(f"{TRAVAIL}/chat/finale"):
    !{LANCER} --config configs/un_chat.yaml 2>&1 | grep --line-buffered -vE 'it/s\\]|examples/s' | tee -a {J}/chat.txt
if not PAUSE:
    apres("chat", f"{TRAVAIL}/chat")
"""),
    md("""
## Étape 4 : tests
Deux passages : **sans pénalité** (comparable au modèle zéro) et **avec pénalité de répétition 1.2**
(contre les boucles). Les catégories **« NON (vraie preuve) »** comptent : jamais vues à l'entraînement.
**flores_rn_fr / flores_fr_rn** : phrases traduites par des professionnels, hors Bible (si FLORES est dans dataset_un).
"""),
    code("""
os.chdir(CODE)
if PAUSE:
    print("Tests sautés : l'entraînement n'est pas fini (version suivante).")
else:
    tests = "tests/dialogues_test.jsonl" + (f" {DATASET_UN}/tests/flores.jsonl" if fait(f"{DATASET_UN}/tests/flores.jsonl") else "")
    !python -u evaluer.py --modele {meilleur_modele()} --tests {tests} --max 100 --montrer 3 \\
        --sortie {J}/resultats_tests.jsonl 2>&1 | tee {J}/notes_tests.txt
    !python -u evaluer.py --modele {meilleur_modele()} --tests {tests} --max 100 --montrer 0 --penalite 1.2 \\
        --sortie {J}/resultats_tests_penalite.jsonl 2>&1 | tee {J}/notes_tests_penalite.txt
"""),
    md("## Étape 5 : essayer le modèle"),
    code("""
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
                         repetition_penalty=1.2, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    reponse = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    if garder:
        historique = f"{prompt} {reponse}<|endoftext|>"
    return reponse

print(dossier)
for q in ["Bonjour", "Bwakeye.", "N'amaki?", "Comment dit-on merci en kirundi ?", "Qui es-tu ?",
          "Traduis en français : Umwana n'umukobwa.", "Que veut dire « Guhodoka » en kirundi ?",
          "Que veut dire « Impingwe » en kirundi ?"]:
    print(q, "->", demander(q, garder=False))
"""),
    md("## Étape 6 : le `.zip` pour le backend\nÀ mettre dans `litgpt/models/local/`, puis `python scripts/convert_local_model.py <dossier>`."),
]

FIN = {
    "kaggle": code("""
if PAUSE:
    print("PAS FINI. Pour continuer : Add Input > Notebooks > un_kaggle (cette version), puis Save & Run All.")
else:
    dossier_modele = meilleur_modele()
    !cd "{dossier_modele}" && zip -qr /kaggle/working/modele_un_fr_rn.zip . -x "etat_entrainement.pt"
    !ls -lh /kaggle/working/modele_un_fr_rn.zip
    print("Le .zip est dans l'onglet Output de cette version.")
print(f"Durée de cette session : {(time.time() - DEBUT) / 3600:.1f} h")
"""),
    "colab": code("""
if PAUSE:
    print("PAS FINI : relance 'Exécuter tout', ça reprend.")
else:
    dossier_modele = meilleur_modele()
    !cd "{dossier_modele}" && zip -qr /content/modele_un_fr_rn.zip . -x "etat_entrainement.pt" && cp /content/modele_un_fr_rn.zip {TRAVAIL}/
    from google.colab import files
    files.download("/content/modele_un_fr_rn.zip")
"""),
    "machine": code("""
if PAUSE:
    print("PAS FINI : relance le notebook, ça reprend.")
else:
    dossier_modele = meilleur_modele()
    !cd "{dossier_modele}" && zip -qr {TRAVAIL}/modele_un_fr_rn.zip . -x "etat_entrainement.pt" && ls -lh {TRAVAIL}/modele_un_fr_rn.zip
"""),
}


def ecrire(nom, debut, plateforme):
    meta = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}}
    if plateforme == "colab":
        meta["accelerator"] = "GPU"
        meta["colab"] = {"gpuType": "T4", "provenance": []}
    nb = {"cells": debut + COMMUN + [FIN[plateforme]], "metadata": meta, "nbformat": 4, "nbformat_minor": 5}
    for dossier in (ICI, ICI / "datasets/dataset_un/notebooks"):
        dossier.mkdir(parents=True, exist_ok=True)
        (dossier / nom).write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Écrit :", nom)


if __name__ == "__main__":
    ecrire("un_kaggle.ipynb", KAGGLE, "kaggle")
    ecrire("un_colab.ipynb", COLAB, "colab")
    ecrire("un_machine.ipynb", MACHINE, "machine")
