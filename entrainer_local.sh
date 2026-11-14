#!/usr/bin/env bash
# entrainer_local.sh : enchaîne tout l'entraînement sur CETTE machine (GPU NVIDIA).
# Équivalent des étapes 3 à 7 du notebook Colab. Chaque étape déjà finie est sautée,
# et une phase interrompue REPREND toute seule au dernier point sauvegardé.
#
#   ./entrainer_local.sh            # tout, dans l'ordre
#   ./entrainer_local.sh phaseA     # une seule étape : phaseA | grow | phaseB | phaseD | tests
#
# Pour le laisser tourner la nuit (même si on ferme le terminal) :
#   nohup ./entrainer_local.sh > entrainement.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"

M=A_METTRE_DANS_DRIVE/modele
J=$M/journaux
mkdir -p "$J"

# Environnement Python : conda "modele_train" (voir environment.yml), sinon .venv.
if [ "${CONDA_DEFAULT_ENV:-}" != "modele_train" ]; then
    BASE_CONDA=$(conda info --base 2>/dev/null || echo "$HOME/anaconda3")
    if [ -f "$BASE_CONDA/etc/profile.d/conda.sh" ] && [ -d "$BASE_CONDA/envs/modele_train" ]; then
        source "$BASE_CONDA/etc/profile.d/conda.sh"
        conda activate modele_train
    elif [ -d .venv ]; then
        source .venv/bin/activate
    fi
fi
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 'Pas de GPU vu par PyTorch : voir README_LOCAL.md, étape 1')"

# Moins de "CUDA out of memory" sur 4 Go : la mémoire est rangée en morceaux extensibles.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

phase() {  # phase <nom>
    if [ -f "$M/$1/finale/config.json" ]; then echo "OK : $1 déjà finie"; return; fi
    echo "=== $1 ($(date '+%F %T'))"
    # systemd-inhibit : l'ordinateur ne se met PAS en veille pendant l'entraînement.
    systemd-inhibit --what=sleep:idle --why="entraînement $1" \
        python -u train.py --config "configs/local/$1.yaml" 2>&1 \
        | grep --line-buffered -vE 'it/s\]|examples/s' | tee -a "$J/$1.txt"
    [ -f "$M/$1/finale/config.json" ] || { echo "ÉCHEC : $1. Lis la fin de $J/$1.txt"; exit 1; }
}

grow() {
    if [ -f "$M/grand_init/config.json" ]; then echo "OK : grand_init existe"; return; fi
    python grow.py --source "$M/phaseA/finale" --sortie "$M/grand_init" \
        --hidden "${HIDDEN:-768}" --layers "${LAYERS:-12}" --heads "${HEADS:-12}"
}

tests() {
    local m
    for m in phaseD/finale phaseD/meilleure phaseB/finale phaseB/meilleure phaseA/finale; do
        [ -f "$M/$m/config.json" ] && break
    done
    python -u evaluer.py --modele "$M/$m" --montrer 3 --sortie "$J/resultats_tests.jsonl" 2>&1 \
        | tee "$J/notes_tests.txt"
}

case "${1:-tout}" in
    phaseA|phaseB|phaseD) phase "$1" ;;
    grow)  grow ;;
    tests) tests ;;
    tout)  phase phaseA; grow; phase phaseB; phase phaseD; tests ;;
    *) echo "Usage : $0 [phaseA|grow|phaseB|phaseD|tests]"; exit 1 ;;
esac
