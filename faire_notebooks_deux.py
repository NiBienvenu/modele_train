#!/usr/bin/env python
"""
faire_notebooks_deux.py : écrit les 3 notebooks du MODÈLE DEUX (suite du modèle un).

    python3 faire_notebooks_deux.py      (appelé aussi par preparer_donnees_deux.py)

    deux_kaggle.ipynb   Kaggle  (Inputs : dataset modele-train, sortie du notebook un_kaggle)
    deux_colab.ipynb    Colab   (Mon Drive : dataset/, dataset_un/, dataset_deux/, modele_un/)
    deux_machine.ipynb  cette machine

Corrections par rapport au un : lecture longue (3 époques, >= 2000 pas, wikipedia_fr_002), Bible réduite,
vocabulaire et cours x5, traductions hors Bible, chat 2 époques + meilleur point + dropout,
pénalité 1.2 + no_repeat_ngram 3, faits Burundi, MÊMES tests que le un et comparaison colonne par colonne.
Une copie va dans datasets/dataset_deux/notebooks/. Ne pas modifier les .ipynb à la main.
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
# Modèle DEUX : français + kirundi ({plateforme})

On **continue le modèle un** (même tokenizer). Nouveau (`dataset_deux`) : grammaire Indimburo, les deux
constitutions (à retenir par cœur), Dusome 5, loi presse 2024, discours, Vision 2040-2060, faits sur le Burundi.

| # | Correction par rapport au modèle un |
|---|---|
| 1 | Lecture **3 époques, au moins 2 000 pas** (~6 h), avec `wikipedia_fr_002` |
| 2 | Bible : **9 000** exemples de chat sur 40 538 |
| 3 | `vocabulaire` x9 (un : x2), `traductions_cours` x15 (un : x3) |
| 4 | Traductions hors Bible : FLORES **dev**, constitution 2018, loi presse |
| 5 | Chat : 2 époques, **meilleur point gardé**, dropout 0.1, weight decay 0.05 |
| 6 | Pénalité de répétition **1.2** + `no_repeat_ngram_size=3` par défaut |
| 7 | Faits simples sur le Burundi (lecture + questions) |
| 8 | **Mêmes tests** que le un, comparés colonne par colonne |

| Étape | Ce qui se passe |
|---|---|
| 0 | Préparer : trouver les dossiers, l'heure de fin de session |
| 1 | Données publiques (Wikipédia FR 001 **et 002**...) |
| 2 | **Lecture longue** |
| 3 | **Discussion** |
| 4 | Tests : modèle un et modèle deux, mêmes réglages, puis comparaison |
| 5 | Essayer le modèle, puis le `.zip` |

**Sauvegardes** : petite sauvegarde tous les 50 pas, **copie complète toutes les 4 h**
(`sauvegarde_4h`, `sauvegarde_8h`), et **pause propre avant la fin de la session**.
Si une étape n'est pas finie à la pause, la **version suivante** du notebook reprend au même pas.
{avant}
"""

KAGGLE = [
    md(INTRO.format(plateforme="Kaggle", avant="""
**Avant de lancer (Inputs)** : **Add Input** 2 fois :
1. Datasets > **modele-train** (dernière version). À sa racine : `dataset/`, `dataset_un/` et `dataset_deux/`.
2. Notebooks > **un_kaggle** (sa sortie contient `modele_un/lecture/finale`, le point de départ, et `chat/finale` pour comparer).
3. (facultatif) Notebooks > **zero_kaggle** : ses `publics/` évitent de retélécharger Wikipédia 001.

**Settings** : Accelerator = **GPU T4 x2**, Internet = **On**. Puis **Save Version > Save & Run All**.

**Session de 12 h** : le notebook s'arrête de lui-même vers 11 h (pause propre), les fichiers restent
dans l'onglet **Output**. **Pour continuer** : Add Input > Notebooks > **ce notebook (deux_kaggle)**,
puis Save & Run All encore une fois : il recopie `modele_deux` et reprend.
La lecture longue (~6 h) + la discussion peuvent demander **2 sessions**.
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
DATASET_UN = os.path.dirname(os.path.dirname(trouver("insiguro/insiguro.txt", "dataset_un")))
DATASET_DEUX = os.path.dirname(os.path.dirname(trouver("ikibonezamvugo/indimburo_yikirundi.txt", "dataset_deux")))
# Le modèle un : sortie du notebook un_kaggle (modele_un/lecture/finale)
UN = os.path.dirname(os.path.dirname(os.path.dirname(trouver(
    "modele_un/lecture/finale/config.json",
    "le modèle un (Add Input > Your Work > Notebooks > un_kaggle)", sauf="modele_deux"))))
TRAVAIL = "/kaggle/working/modele_deux"
CODE = "/kaggle/working/code_deux"
shutil.copytree(f"{DATASET_DEUX}/code", CODE, dirs_exist_ok=True)

# Reprise : sortie d'une version précédente de CE notebook ajoutée en Input
if not os.path.exists(TRAVAIL):
    anciens = sorted(glob.glob("/kaggle/input/**/modele_deux/lecture", recursive=True))
    if anciens:
        print("Reprise depuis", os.path.dirname(anciens[0]), "(copie, quelques minutes)")
        shutil.copytree(os.path.dirname(anciens[0]), TRAVAIL)
# Wikipédia : publics/ du zéro s'il a déjà 001 ET 002, sinon téléchargés dans TRAVAIL
_pub = sorted(glob.glob("/kaggle/input/**/publics/lecture_fr/wikipedia_fr_002.txt", recursive=True))
PUBLICS = os.path.dirname(os.path.dirname(_pub[0])) if _pub else f"{TRAVAIL}/publics"
"""),
]

COLAB = [
    md(INTRO.format(plateforme="Google Colab", avant="""
**Avant de lancer** : dans **Mon Drive** (à la racine) il faut :
1. `dataset/`, `dataset_un/` et `dataset_deux/` (les dossiers `datasets/...` de la machine)
2. `modele_un/lecture/finale/` (point de départ) et `modele_un/chat/finale/` (pour comparer les tests).
   Sur Kaggle : onglet Output du notebook un_kaggle > télécharger, puis le mettre dans Drive.

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
DATASET_DEUX = "/content/drive/MyDrive/dataset_deux"
UN = "/content/drive/MyDrive/modele_un"
TRAVAIL = "/content/drive/MyDrive/modele_deux"    # écriture dans Drive (survit aux coupures)
for d, quoi in [(f"{DATASET}/dialogues", "dataset"), (f"{DATASET_UN}/insiguro", "dataset_un"),
                (f"{DATASET_DEUX}/ikibonezamvugo", "dataset_deux"),
                (f"{UN}/lecture/finale/config.json", "modele_un/lecture/finale")]:
    if not os.path.exists(d):
        raise SystemExit(f"Manque dans Mon Drive : {quoi} (voir en haut).")
CODE = "/content/code_deux"
shutil.copytree(f"{DATASET_DEUX}/code", CODE, dirs_exist_ok=True)
!pip install -q -r {CODE}/requirements.txt
PUBLICS = "/content/drive/MyDrive/publics"
"""),
]

MACHINE = [
    md(INTRO.format(plateforme="sur cette machine", avant="""
**Avant de lancer** : ouvre ce notebook depuis `modele_train/`, dans l'environnement PyTorch + CUDA
(`README_LOCAL.md`). Il faut `datasets/dataset`, `datasets/dataset_un`, `datasets/dataset_deux` et le modèle un
dans `modeles/modele_un/lecture/finale` (+ `chat/finale` pour comparer), depuis l'onglet Output de un_kaggle.
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
DATASET_DEUX = f"{ICI}/datasets/dataset_deux"
UN = f"{ICI}/modeles/modele_un"
TRAVAIL = f"{ICI}/modeles/modele_deux"
for d, quoi in [(f"{DATASET}/dialogues", "dataset"), (f"{DATASET_UN}/insiguro", "dataset_un"),
                (f"{DATASET_DEUX}/ikibonezamvugo", "dataset_deux (preparer_donnees_deux.py)"),
                (f"{UN}/lecture/finale/config.json", "modele_un/lecture/finale")]:
    if not os.path.exists(d):
        raise SystemExit(f"Manque : {quoi}.")
CODE = f"{TRAVAIL}/code"
shutil.copytree(f"{DATASET_DEUX}/code", CODE, dirs_exist_ok=True)
if not torch.cuda.is_available():
    raise SystemExit("PyTorch ne voit pas de GPU NVIDIA (README_LOCAL.md).")
go = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU : {torch.cuda.get_device_name(0)}, {go:.0f} Go")
if go < 10:   # petite carte : mêmes calculs, par plus petits morceaux
    for nom, lot, acc in [("deux_lecture", 1, 128), ("deux_chat", 1, 32)]:
        chemin = f"{CODE}/configs/{nom}.yaml"
        texte = open(chemin, encoding="utf-8").read()
        texte = re.sub(r"(?m)^taille_lot: .*", f"taille_lot: {lot}", texte)
        texte = re.sub(r"(?m)^accumulation: .*", f"accumulation: {acc}", texte)
        texte += "\\ngradient_checkpointing: true\\n"
        open(chemin, "w", encoding="utf-8").write(texte)
    print("Configs adaptées à une petite carte (beaucoup plus lent qu'un T4).")
PUBLICS = f"{ICI}/modeles/publics"
"""),
]

COMMUN = [
    code("""
# 0.b  Commun : variables lues par train.py, heure de fin, état de chaque étape
import torch
os.makedirs(f"{TRAVAIL}/journaux", exist_ok=True)
J = f"{TRAVAIL}/journaux"
os.environ.update(DATASET=DATASET, DATASET_UN=DATASET_UN, DATASET_DEUX=DATASET_DEUX, UN=UN, TRAVAIL=TRAVAIL,
                  PUBLICS=PUBLICS)
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
    return f"{UN}/lecture/finale"

def modele_un():
    for sous in ("chat/finale", "chat/meilleure", "lecture/finale"):
        if fait(f"{UN}/{sous}/config.json"):
            return f"{UN}/{sous}"

print("DATASET    =", DATASET); print("DATASET_UN =", DATASET_UN); print("DATASET_DEUX =", DATASET_DEUX)
print("UN         =", UN)
print("PUBLICS    =", PUBLICS); print("TRAVAIL    =", TRAVAIL)
for nom, ph in [("Lecture", "lecture"), ("Discussion", "chat")]:
    d = f"{TRAVAIL}/{ph}"
    print(f"{nom:11s}: " + ("FINI" if fait(f"{d}/finale") else "en cours (reprise)" if fait(f"{d}/derniere")
                             else "pas commencé"))
"""),
    md("## Étape 1 : données publiques\nIl faut **deux** fichiers Wikipédia (001 et 002, 20 M de mots chacun) : correction 1."),
    code("""
os.chdir(CODE)
if not glob.glob(f"{PUBLICS}/lecture_fr/wikipedia_fr_002.txt"):
    !python -u collecte/telecharger_publics.py --sortie {PUBLICS} --mots 40000000 2>&1 | grep -v "examples/s" | tee {J}/publics.txt
!ls {PUBLICS}/lecture_fr | head -3; ls {PUBLICS}/dialogues
"""),
    md("""
## Étape 2 : lecture longue (continue le modèle un)
3 époques, au moins 2 000 pas (~6 h sur 2 x T4). Si la session finit avant, pause propre et reprise
à la version suivante. Les phrases test montrent la constitution (« Ingingo ya 5 ») et la grammaire.
"""),
    code("""
os.chdir(CODE)
if not PAUSE and not fait(f"{TRAVAIL}/lecture/finale"):
    !{LANCER} --config configs/deux_lecture.yaml 2>&1 | grep --line-buffered -vE 'it/s\\]|examples/s' | tee -a {J}/lecture.txt
apres("lecture", f"{TRAVAIL}/lecture")
"""),
    md("## Étape 3 : discussion (2 époques, le meilleur point est gardé)"),
    code("""
os.chdir(CODE)
if not PAUSE and not fait(f"{TRAVAIL}/chat/finale"):
    !{LANCER} --config configs/deux_chat.yaml 2>&1 | grep --line-buffered -vE 'it/s\\]|examples/s' | tee -a {J}/chat.txt
if not PAUSE:
    apres("chat", f"{TRAVAIL}/chat")
"""),
    md("""
## Étape 4 : tests (correction 8)
**Exactement les mêmes tests** que le modèle un (`dialogues_test.jsonl` + les 600 FLORES devtest, jamais vus).
On teste le **modèle un ET le modèle deux** avec les mêmes réglages, puis on compare colonne par colonne :
1. réglage par défaut (pénalité 1.2, pas de répétition de 3 mots) ;
2. sans pénalité (1.0, comme les anciens résultats du un).
`+` = mieux, `-` = moins bien, `=` = pareil (écart de chrF).
"""),
    code("""
os.chdir(CODE)
if PAUSE:
    print("Tests sautés : l'entraînement n'est pas fini (version suivante).")
else:
    tests = f"tests/dialogues_test.jsonl {DATASET_DEUX}/tests/flores.jsonl"
    for nom, modele in [("un", modele_un()), ("deux", meilleur_modele())]:
        print("=====", nom, modele)
        !python -u evaluer.py --modele {modele} --tests {tests} --max 100 --montrer 3 \\
            --sortie {J}/resultats_{nom}.jsonl 2>&1 | tee {J}/notes_{nom}.txt
        !python -u evaluer.py --modele {modele} --tests {tests} --max 100 --montrer 0 --penalite 1.0 --sans-repeter 0 \\
            --sortie {J}/resultats_{nom}_sans_penalite.jsonl 2>&1 | tee {J}/notes_{nom}_sans_penalite.txt
    print("\\n##### COMPARAISON (pénalité 1.2 + no_repeat 3)")
    !python comparer_resultats.py {J}/resultats_un.jsonl {J}/resultats_deux.jsonl --noms un deux | tee {J}/comparaison.txt
    print("\\n##### COMPARAISON (sans pénalité)")
    !python comparer_resultats.py {J}/resultats_un_sans_penalite.jsonl {J}/resultats_deux_sans_penalite.jsonl --noms un deux | tee -a {J}/comparaison.txt
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
                         repetition_penalty=1.2, no_repeat_ngram_size=3, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    reponse = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    if garder:
        historique = f"{prompt} {reponse}<|endoftext|>"
    return reponse

print(dossier)
for q in ["Bonjour", "Bwakeye.", "N'amaki?", "Où se trouve le Burundi ?", "Quels sont les pays voisins du Burundi ?",
          "Traduis en français : Umwana n'umukobwa.", "Que veut dire « Guhodoka » en kirundi ?",
          "Ingingo ya 5 y'Ibwirizwa shingiro rya 2018 ivuga iki?", "Que signifie « karangura » en grammaire kirundi ?",
          "Mbwira ingingo ya 26 y'Ibwirizwa shingiro rya 2018 uko yanditse."]:
    print(q, "->", demander(q, garder=False))
"""),
    md("## Étape 6 : le `.zip` pour le backend\nÀ mettre dans `litgpt/models/local/`, puis `python scripts/convert_local_model.py <dossier>`."),
]

FIN = {
    "kaggle": code("""
if PAUSE:
    print("PAS FINI. Pour continuer : Add Input > Notebooks > deux_kaggle (cette version), puis Save & Run All.")
else:
    dossier_modele = meilleur_modele()
    !cd "{dossier_modele}" && zip -qr /kaggle/working/modele_deux_fr_rn.zip . -x "etat_entrainement.pt"
    !ls -lh /kaggle/working/modele_deux_fr_rn.zip
    print("Le .zip est dans l'onglet Output de cette version.")
print(f"Durée de cette session : {(time.time() - DEBUT) / 3600:.1f} h")
"""),
    "colab": code("""
if PAUSE:
    print("PAS FINI : relance 'Exécuter tout', ça reprend.")
else:
    dossier_modele = meilleur_modele()
    !cd "{dossier_modele}" && zip -qr /content/modele_deux_fr_rn.zip . -x "etat_entrainement.pt" && cp /content/modele_deux_fr_rn.zip {TRAVAIL}/
    from google.colab import files
    files.download("/content/modele_deux_fr_rn.zip")
"""),
    "machine": code("""
if PAUSE:
    print("PAS FINI : relance le notebook, ça reprend.")
else:
    dossier_modele = meilleur_modele()
    !cd "{dossier_modele}" && zip -qr {TRAVAIL}/modele_deux_fr_rn.zip . -x "etat_entrainement.pt" && ls -lh {TRAVAIL}/modele_deux_fr_rn.zip
"""),
}


def ecrire(nom, debut, plateforme):
    meta = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}}
    if plateforme == "colab":
        meta["accelerator"] = "GPU"
        meta["colab"] = {"gpuType": "T4", "provenance": []}
    nb = {"cells": debut + COMMUN + [FIN[plateforme]], "metadata": meta, "nbformat": 4, "nbformat_minor": 5}
    for dossier in (ICI, ICI / "datasets/dataset_deux/notebooks"):
        dossier.mkdir(parents=True, exist_ok=True)
        (dossier / nom).write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Écrit :", nom)


if __name__ == "__main__":
    ecrire("deux_kaggle.ipynb", KAGGLE, "kaggle")
    ecrire("deux_colab.ipynb", COLAB, "colab")
    ecrire("deux_machine.ipynb", MACHINE, "machine")
