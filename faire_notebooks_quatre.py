#!/usr/bin/env python
"""
faire_notebooks_quatre.py : écrit les 3 notebooks du MODÈLE QUATRE (suite du modèle trois).

    python3 faire_notebooks_quatre.py      (appelé aussi par preparer_donnees_quatre.py)

Fait pour une session Kaggle de 12 h : ~5 h d'entraînement (lecture ~2 h + discussion ~2,5 h), puis tests.
Une copie va dans datasets/dataset_quatre/notebooks/. Ne pas modifier les .ipynb à la main.
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
# Modèle QUATRE : français + kirundi ({plateforme})

On **continue la lecture du modèle trois** (même tokenizer). Pas de nouveau kirundi : on corrige ce que le run trois a montré.

| # | Correction par rapport au modèle trois |
|---|---|
| 1 | Bible **24 000** passages dans `dataset_quatre` (trois : 9 000, vieux dataset sur Kaggle), x2 |
| 2 | **Contrôle des fichiers** au départ (`attendu.json`) : arrêt si le dataset Kaggle n'est pas à jour |
| 3 | **Évaluation honnête** : mise de côté AVANT les répétitions, même mesure pendant et à la fin |
| 4 | Lecture **2 époques** (~2 h au lieu de 7 h), Wikipédia FR réduit (12 000 paragraphes) |
| 5 | Répétitions baissées (x50/x40/x30 -> x10), plus de traduction (FLORES dev x6, fr_rn x4), 6 époques |
| 6 | Fin de phrase gardée quand un exemple est coupé ; plus de réponse vide aux tests |
| 7 | Comparaison trois / quatre **sur tous les tests** (l'historique compte) |

| Étape | Ce qui se passe |
|---|---|
| 0 | Préparer : trouver les dossiers, l'heure de fin de session |
| 1 | Données publiques (Wikipédia FR 001 à 004, ~20 min) |
| 2 | **Lecture** (~2 h) |
| 3 | **Discussion** (~2,5 h) |
| 4 | Tests : modèle trois et modèle quatre, puis comparaison |
| 5 | Essayer le modèle, puis le `.zip` |

**Sauvegardes** : petite sauvegarde tous les 50 pas, copie complète toutes les 4 h,
et **pause propre avant la fin de la session**. Si ce n'est pas fini, la **version suivante** reprend au même pas.
{avant}
"""

KAGGLE = [
    md(INTRO.format(plateforme="Kaggle", avant="""
**Avant de lancer (Inputs)** : **Add Input** 2 fois :
1. Datasets > **modele-train** (**nouvelle version** avec `dataset_quatre/`). À sa racine : `dataset/`, `dataset_un/`, `dataset_deux/`, `dataset_trois/` et `dataset_quatre/`.
2. Notebooks > **trois_kaggle** (sa sortie contient `modele_trois/lecture/finale`, le point de départ, et `chat/finale` pour comparer).

**Settings** : Accelerator = **GPU T4 x2**, Internet = **On**. Puis **Save Version > Save & Run All**.

**Session de 12 h** : le notebook s'arrête de lui-même vers 11 h (pause propre), les fichiers restent
dans l'onglet **Output**. Prévu : tout tient dans **une** session (~6 h avec les tests).
Si ça ne tient pas : Add Input > Notebooks > **ce notebook (quatre_kaggle)**, puis Save & Run All : il reprend.
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
DATASET_TROIS = os.path.dirname(os.path.dirname(trouver("lecture_rn/dusome4.txt", "dataset_trois")))
DATASET_QUATRE = os.path.dirname(os.path.dirname(trouver("traduction/bible_quatre.jsonl",
                                                         "dataset_quatre (nouvelle version de modele-train)")))
# Le modèle trois : sortie du notebook trois_kaggle (modele_trois/lecture/finale)
TROIS = os.path.dirname(os.path.dirname(os.path.dirname(trouver(
    "modele_trois/lecture/finale/config.json",
    "le modèle trois (Add Input > Your Work > Notebooks > trois_kaggle)", sauf="modele_quatre"))))
TRAVAIL = "/kaggle/working/modele_quatre"
CODE = "/kaggle/working/code_quatre"
shutil.copytree(f"{DATASET_QUATRE}/code", CODE, dirs_exist_ok=True)

# Reprise : sortie d'une version précédente de CE notebook ajoutée en Input
if not os.path.exists(TRAVAIL):
    anciens = sorted(glob.glob("/kaggle/input/**/modele_quatre/lecture", recursive=True))
    if anciens:
        print("Reprise depuis", os.path.dirname(anciens[0]), "(copie, quelques minutes)")
        shutil.copytree(os.path.dirname(anciens[0]), TRAVAIL)
# Wikipédia 001 à 004 : déjà dans un Input s'il existe, sinon téléchargés dans TRAVAIL
_pub = sorted(glob.glob("/kaggle/input/**/publics/lecture_fr/wikipedia_fr_004.txt", recursive=True))
PUBLICS = os.path.dirname(os.path.dirname(_pub[0])) if _pub else f"{TRAVAIL}/publics"
"""),
]

COLAB = [
    md(INTRO.format(plateforme="Google Colab", avant="""
**Avant de lancer** : dans **Mon Drive** (à la racine) il faut :
1. `dataset/`, `dataset_un/`, `dataset_deux/`, `dataset_trois/` et `dataset_quatre/` (les dossiers `datasets/...` de la machine)
2. `modele_trois/lecture/finale/` (point de départ) et `modele_trois/chat/finale/` (pour comparer les tests).
   Sur Kaggle : onglet Output du notebook trois_kaggle > télécharger, puis le mettre dans Drive.

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
DATASET_TROIS = "/content/drive/MyDrive/dataset_trois"
DATASET_QUATRE = "/content/drive/MyDrive/dataset_quatre"
TROIS = "/content/drive/MyDrive/modele_trois"
TRAVAIL = "/content/drive/MyDrive/modele_quatre"    # écriture dans Drive (survit aux coupures)
for d, quoi in [(f"{DATASET}/dialogues", "dataset"), (f"{DATASET_UN}/insiguro", "dataset_un"),
                (f"{DATASET_DEUX}/ikibonezamvugo", "dataset_deux"), (f"{DATASET_TROIS}/lecture_rn", "dataset_trois"),
                (f"{DATASET_QUATRE}/traduction", "dataset_quatre"),
                (f"{TROIS}/lecture/finale/config.json", "modele_trois/lecture/finale")]:
    if not os.path.exists(d):
        raise SystemExit(f"Manque dans Mon Drive : {quoi} (voir en haut).")
CODE = "/content/code_quatre"
shutil.copytree(f"{DATASET_QUATRE}/code", CODE, dirs_exist_ok=True)
!pip install -q -r {CODE}/requirements.txt
PUBLICS = "/content/drive/MyDrive/publics"
"""),
]

MACHINE = [
    md(INTRO.format(plateforme="sur cette machine", avant="""
**Avant de lancer** : ouvre ce notebook depuis `modele_train/`, dans l'environnement PyTorch + CUDA
(`README_LOCAL.md`). Il faut `datasets/dataset`, `datasets/dataset_un`, `datasets/dataset_deux` et le modèle un
+ `datasets/dataset_trois` et `datasets/dataset_quatre` (preparer_donnees_quatre.py), et le modèle trois dans `modeles/modele_trois/lecture/finale` (+ `chat/finale`).
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
DATASET_TROIS = f"{ICI}/datasets/dataset_trois"
DATASET_QUATRE = f"{ICI}/datasets/dataset_quatre"
TROIS = f"{ICI}/modeles/modele_trois"
TRAVAIL = f"{ICI}/modeles/modele_quatre"
for d, quoi in [(f"{DATASET}/dialogues", "dataset"), (f"{DATASET_UN}/insiguro", "dataset_un"),
                (f"{DATASET_DEUX}/ikibonezamvugo", "dataset_deux"),
                (f"{DATASET_TROIS}/lecture_rn", "dataset_trois"),
                (f"{DATASET_QUATRE}/traduction", "dataset_quatre (preparer_donnees_quatre.py)"),
                (f"{TROIS}/lecture/finale/config.json", "modele_trois/lecture/finale")]:
    if not os.path.exists(d):
        raise SystemExit(f"Manque : {quoi}.")
CODE = f"{TRAVAIL}/code"
shutil.copytree(f"{DATASET_QUATRE}/code", CODE, dirs_exist_ok=True)
if not torch.cuda.is_available():
    raise SystemExit("PyTorch ne voit pas de GPU NVIDIA (README_LOCAL.md).")
go = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU : {torch.cuda.get_device_name(0)}, {go:.0f} Go")
if go < 10:   # petite carte : mêmes calculs, par plus petits morceaux
    for nom, lot, acc in [("quatre_lecture", 1, 128), ("quatre_chat", 1, 32)]:
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
os.environ.update(DATASET=DATASET, DATASET_UN=DATASET_UN, DATASET_DEUX=DATASET_DEUX, DATASET_TROIS=DATASET_TROIS,
                  DATASET_QUATRE=DATASET_QUATRE, TROIS=TROIS, TRAVAIL=TRAVAIL,
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
    return f"{TROIS}/lecture/finale"

def modele_trois():
    for sous in ("chat/finale", "chat/meilleure", "lecture/finale"):
        if fait(f"{TROIS}/{sous}/config.json"):
            return f"{TROIS}/{sous}"

print("DATASET    =", DATASET); print("DATASET_UN =", DATASET_UN); print("DATASET_DEUX =", DATASET_DEUX)
print("DATASET_TROIS =", DATASET_TROIS); print("DATASET_QUATRE =", DATASET_QUATRE); print("TROIS       =", TROIS)
print("PUBLICS    =", PUBLICS); print("TRAVAIL    =", TRAVAIL)
for nom, ph in [("Lecture", "lecture"), ("Discussion", "chat")]:
    d = f"{TRAVAIL}/{ph}"
    print(f"{nom:11s}: " + ("FINI" if fait(f"{d}/finale") else "en cours (reprise)" if fait(f"{d}/derniere")
                             else "pas commencé"))
"""),
    code("""
# 0.c  Contrôle : chaque fichier a-t-il le nombre de lignes prévu ? (run trois : la Bible avait 9 000 lignes
#      sur Kaggle au lieu de 16 000, personne ne l'a vu). Sinon : arrêt tout de suite, pas 10 h plus tard.
import json
RACINES = {"dataset": DATASET, "dataset_un": DATASET_UN, "dataset_deux": DATASET_DEUX,
           "dataset_trois": DATASET_TROIS, "dataset_quatre": DATASET_QUATRE}
erreurs = []
for f, n in json.load(open(f"{DATASET_QUATRE}/attendu.json", encoding="utf-8")).items():
    racine, reste = f.split("/", 1)
    chemin = f"{RACINES[racine]}/{reste}"
    vu = sum(1 for l in open(chemin, encoding="utf-8") if l.strip()) if os.path.exists(chemin) else 0
    print(f"{'OK ' if vu == n else 'ERR'} {vu:7d} / {n:7d}  {f}")
    if vu != n:
        erreurs.append(f)
if erreurs:
    raise SystemExit("Dataset pas à jour : " + ", ".join(erreurs) + ". Refais une version du dataset modele-train.")
"""),
    md("## Étape 1 : données publiques\n**Quatre** fichiers Wikipédia (001 à 004, 20 M de mots chacun), pour moins répéter les mêmes textes."),
    code("""
os.chdir(CODE)
if not glob.glob(f"{PUBLICS}/lecture_fr/wikipedia_fr_004.txt"):
    !python -u collecte/telecharger_publics.py --sortie {PUBLICS} --mots 80000000 2>&1 | grep -v "examples/s" | tee {J}/publics.txt
!ls {PUBLICS}/lecture_fr | head -3; ls {PUBLICS}/dialogues
"""),
    md("""
## Étape 2 : lecture (continue la lecture du modèle trois)
2 époques (~2 h sur 2 x T4), le meilleur point est gardé. Si la session finit avant, pause propre et reprise.
"""),
    code("""
os.chdir(CODE)
if not PAUSE and not fait(f"{TRAVAIL}/lecture/finale"):
    !{LANCER} --config configs/quatre_lecture.yaml 2>&1 | grep --line-buffered -vE 'it/s\\]|examples/s' | tee -a {J}/lecture.txt
apres("lecture", f"{TRAVAIL}/lecture")
"""),
    md("## Étape 3 : discussion (6 époques, ~2,5 h, le meilleur point est gardé)"),
    code("""
os.chdir(CODE)
if not PAUSE and not fait(f"{TRAVAIL}/chat/finale"):
    !{LANCER} --config configs/quatre_chat.yaml 2>&1 | grep --line-buffered -vE 'it/s\\]|examples/s' | tee -a {J}/chat.txt
if not PAUSE:
    apres("chat", f"{TRAVAIL}/chat")
"""),
    md("""
## Étape 4 : tests
**Exactement les mêmes tests** que le deux et le trois (`dialogues_test.jsonl` + FLORES devtest, jamais vus).
Une seule passe, **sans pénalité** (le run deux a montré que c'est mieux). Puis comparaison trois / quatre.
`+` = mieux, `-` = moins bien, `=` = pareil (écart de chrF).
"""),
    code("""
os.chdir(CODE)
if PAUSE:
    print("Tests sautés : l'entraînement n'est pas fini (version suivante).")
else:
    tests = "tests/dialogues_test.jsonl tests/flores.jsonl"
    for nom, modele in [("trois", modele_trois()), ("quatre", meilleur_modele())]:
        print("=====", nom, modele)
        !python -u evaluer.py --modele {modele} --tests {tests} --max 100 --montrer 3 \\
            --sortie {J}/resultats_{nom}.jsonl 2>&1 | tee {J}/notes_{nom}.txt
    print("\\n##### COMPARAISON trois / quatre (sans pénalité)")
    !python comparer_resultats.py {J}/resultats_trois.jsonl {J}/resultats_quatre.jsonl --noms trois quatre | tee {J}/comparaison.txt
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
                         repetition_penalty=1.05, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
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
    print("PAS FINI. Pour continuer : Add Input > Notebooks > quatre_kaggle (cette version), puis Save & Run All.")
else:
    dossier_modele = meilleur_modele()
    !cd "{dossier_modele}" && zip -qr /kaggle/working/modele_quatre_fr_rn.zip . -x "etat_entrainement.pt"
    !ls -lh /kaggle/working/modele_quatre_fr_rn.zip
    print("Le .zip est dans l'onglet Output de cette version.")
print(f"Durée de cette session : {(time.time() - DEBUT) / 3600:.1f} h")
"""),
    "colab": code("""
if PAUSE:
    print("PAS FINI : relance 'Exécuter tout', ça reprend.")
else:
    dossier_modele = meilleur_modele()
    !cd "{dossier_modele}" && zip -qr /content/modele_quatre_fr_rn.zip . -x "etat_entrainement.pt" && cp /content/modele_quatre_fr_rn.zip {TRAVAIL}/
    from google.colab import files
    files.download("/content/modele_quatre_fr_rn.zip")
"""),
    "machine": code("""
if PAUSE:
    print("PAS FINI : relance le notebook, ça reprend.")
else:
    dossier_modele = meilleur_modele()
    !cd "{dossier_modele}" && zip -qr {TRAVAIL}/modele_quatre_fr_rn.zip . -x "etat_entrainement.pt" && ls -lh {TRAVAIL}/modele_quatre_fr_rn.zip
"""),
}


def ecrire(nom, debut, plateforme):
    meta = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}}
    if plateforme == "colab":
        meta["accelerator"] = "GPU"
        meta["colab"] = {"gpuType": "T4", "provenance": []}
    nb = {"cells": debut + COMMUN + [FIN[plateforme]], "metadata": meta, "nbformat": 4, "nbformat_minor": 5}
    for dossier in (ICI, ICI / "datasets/dataset_quatre/notebooks"):
        dossier.mkdir(parents=True, exist_ok=True)
        (dossier / nom).write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Écrit :", nom)


if __name__ == "__main__":
    ecrire("quatre_kaggle.ipynb", KAGGLE, "kaggle")
    ecrire("quatre_colab.ipynb", COLAB, "colab")
    ecrire("quatre_machine.ipynb", MACHINE, "machine")
