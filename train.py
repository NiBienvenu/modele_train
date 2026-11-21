#!/usr/bin/env python
"""
train.py : le script d'entraînement unique pour toutes les phases.

    Phase A : grammaire, conjugaison, dictionnaires   (mode: pretrain)
    Phase B : textes généraux + rappel de grammaire   (mode: pretrain)
    Phase C : lois, décrets + rappel de grammaire     (mode: pretrain)
    Phase D : questions / réponses                    (mode: sft)

Utilisation dans Colab :
    !python train.py --config configs/phaseA.yaml

Tout ce qui change d'une phase à l'autre est dans le fichier YAML.
Le code, lui, reste le même.

Plusieurs GPU (Kaggle T4 x2) :
    !torchrun --nproc_per_node 2 train.py --config configs/un_lecture.yaml
Chaque GPU lit une moitié différente des données ; le résultat est le même qu'avec un GPU
(même taille de pas), mais environ 2 fois plus vite.

Sessions limitées (Kaggle 12 h) :
    FIN_SESSION = heure de fin (secondes Unix), donnée par le notebook.
    Avant cette heure (moins "marge_minutes"), on sauvegarde et on s'arrête proprement (PAUSE) :
    la version suivante du notebook reprend au même pas.
    "sauver_toutes_les_heures: 4" garde en plus une copie du modèle toutes les 4 h (sauvegarde_4h, 8h...).
"""

import argparse
import itertools
import math
import os
import random
import time

import torch
import yaml
from datasets import concatenate_datasets, load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)


# ---------------------------------------------------------------------------
# 0. Petits outils
# ---------------------------------------------------------------------------

def charger_config(chemin):
    """
    Lit le fichier YAML de la phase.

    {DATASET} et {TRAVAIL} dans le YAML sont remplacés par les variables
    d'environnement du même nom. Le même YAML marche donc partout :
        Colab   : DATASET=/content/drive/MyDrive/dataset   TRAVAIL=/content/drive/MyDrive/modele_zero
        Kaggle  : DATASET=/kaggle/input/...                TRAVAIL=/kaggle/working/modele_zero
        machine : DATASET=datasets/dataset                 TRAVAIL=modeles/modele_zero
    """
    with open(chemin, encoding="utf-8") as f:
        texte = f.read()
    import re
    # {DATASET}, {TRAVAIL}, et pour le modèle un : {DATASET_UN}, {ZERO} (modèle zéro), {PUBLICS}
    for nom in sorted(set(re.findall(r"\{([A-Z_]+)\}", texte))):
        if not os.environ.get(nom):
            raise SystemExit(f"La variable {nom} n'est pas définie (voir la cellule 0 du notebook).")
        texte = texte.replace("{" + nom + "}", os.environ[nom].rstrip("/"))
    return yaml.safe_load(texte)


def fixer_hasard(graine):
    """Même graine = mêmes résultats. Utile pour comparer deux essais."""
    random.seed(graine)
    torch.manual_seed(graine)
    torch.cuda.manual_seed_all(graine)


RANG, NB_GPU = 0, 1   # rang de ce processus et nombre de processus (plusieurs GPU)


def log(msg):
    if RANG == 0:  # un seul GPU écrit le journal
        print(time.strftime("[%H:%M:%S] "), msg, flush=True)


def init_distribue():
    """Lancé par torchrun : un processus par GPU. Sinon : un seul processus."""
    global RANG, NB_GPU
    if int(os.environ.get("WORLD_SIZE", "1")) <= 1:
        return None
    import torch.distributed as dist
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    RANG, NB_GPU = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
    return local


def attendre_tous():
    if NB_GPU > 1:
        torch.distributed.barrier()


# ---------------------------------------------------------------------------
# 1. MÉCANISME 1 : LE TOKENIZER (couper le texte en numéros)
# ---------------------------------------------------------------------------

def charger_tokenizer(chemin):
    """
    Charge NOTRE tokenizer français + kirundi (créé par train_tokenizer.py).
    Il transforme "Abana baragiye" en une liste de numéros, par exemple
    [812, 1045, 3301, 7720]. Le modèle ne voit jamais de lettres, seulement
    ces numéros.
    """
    tok = AutoTokenizer.from_pretrained(chemin)

    # Le "pad" sert à remplir les phrases courtes pour qu'elles aient toutes
    # la même longueur dans un paquet (batch).
    if tok.pad_token is None:
        if "<pad>" in tok.get_vocab():
            tok.pad_token = "<pad>"
        else:
            tok.pad_token = tok.eos_token
    return tok


# ---------------------------------------------------------------------------
# 2. LES DONNÉES
# ---------------------------------------------------------------------------

def charger_sources(sources, format_fichier, decoupage="paragraph", part_eval=0.0):
    """
    Charge chaque source de données et la répète autant de fois que demandé.

    Exemple dans le YAML :
        sources:
          - fichiers: /content/drive/MyDrive/donnees/conjugaison_rn/*.txt
            repeter: 4      # le kirundi est rare, on le montre 4 fois
          - fichiers: /content/drive/MyDrive/donnees/grammaire_fr/*.txt
            repeter: 1

    "repeter" permet d'équilibrer les langues et de garder 15 à 20 % de
    grammaire dans les phases B et C (pour que le modèle n'oublie pas).
    """
    import glob

    morceaux, morceaux_eval = [], []
    for src in sources:
        # Une source absente (ex : PIAF pas téléchargé) est sautée, avec un avertissement
        # On résout le motif ici (ex : wikipedia_fr_00[12].txt, [!b]*.txt) : liste exacte des fichiers
        fichiers = sorted(glob.glob(src["fichiers"]))
        if not fichiers:
            log(f"ATTENTION : aucun fichier pour {src['fichiers']} (source sautée)")
            continue
        if format_fichier == "text":
            # sample_by="paragraph" : un exemple = un paragraphe entier
            # (séparé par une ligne vide). Le modèle garde ainsi le contexte.
            ds = load_dataset("text", data_files=fichiers,
                              sample_by=decoupage, split="train")
            ds = ds.filter(lambda ex: len(ex["text"].strip()) > 0)
        else:
            # Pour la phase D : fichiers .jsonl avec {"user": ..., "assistant": ...}
            ds = load_dataset("json", data_files=fichiers, split="train")
            # Tout en "messages" (role, content) : les fichiers peuvent avoir d'autres colonnes
            # (source, page...) ou l'autre format {"user","assistant"} ; sinon on ne peut pas les mélanger.
            def en_messages(ex):
                m = ex.get("messages") or [{"role": "user", "content": ex["user"]},
                                           {"role": "assistant", "content": ex["assistant"]}]
                return {"messages": [{"role": x["role"], "content": x["content"]} for x in m]}
            ds = ds.map(en_messages, remove_columns=ds.column_names)

        # "max_exemples" : ne garder qu'un tirage au hasard (ex : 9 000 versets sur 40 538)
        if src.get("max_exemples") and len(ds) > src["max_exemples"]:
            ds = ds.shuffle(seed=42).select(range(int(src["max_exemples"])))
        # "minimum" : nombre d'exemples attendu. Moins = le dataset Kaggle n'est pas à jour -> on s'arrête.
        # (run trois : bible_choisie avait 9 000 lignes sur Kaggle au lieu de 16 000, sans que personne le voie)
        if src.get("minimum") and len(ds) < int(src["minimum"]):
            raise SystemExit(f"ERREUR : {src['fichiers']} a {len(ds)} exemples, il en faut au moins "
                             f"{src['minimum']}. Mets à jour le dataset (Kaggle : nouvelle version).")
        n = int(src.get("repeter", 1))
        # part_eval > 0 : on met de côté l'évaluation de CETTE source AVANT de la répéter.
        # Sinon des copies de la même phrase sont à la fois dans l'entraînement et dans l'évaluation,
        # et la loss d'évaluation mesure le par cœur (run trois).
        n_eval = 0
        if part_eval and len(ds) >= 20:
            coupe = ds.train_test_split(test_size=part_eval, seed=42)
            ds, ds_ev = coupe["train"], coupe["test"]
            morceaux_eval.append(ds_ev)
            n_eval = len(ds_ev)
        log(f"Source {src['fichiers']} : {len(ds)} exemples x {n}"
            + (f" (+ {n_eval} pour l'évaluation)" if n_eval else ""))
        morceaux.extend([ds] * n)

    # On mélange tout pour que le modèle ne voie pas une seule langue
    # pendant des heures, puis l'autre.
    train = concatenate_datasets(morceaux).shuffle(seed=42)
    if not part_eval:
        return train
    return train, concatenate_datasets(morceaux_eval).shuffle(seed=42)


def preparer_pretrain(ds, tok, longueur):
    """
    Mode "pretrain" (phases A, B, C).

    1) On transforme chaque texte en numéros, et on ajoute <|endoftext|> à la
       fin pour que le modèle sache où un document se termine.
    2) On colle tous les numéros bout à bout, puis on coupe en blocs de
       taille fixe (par exemple 512). Aucun espace n'est gaspillé en "pad".
    """
    def tokeniser(lot):
        return tok([t + tok.eos_token for t in lot["text"]])

    ds = ds.map(tokeniser, batched=True, remove_columns=ds.column_names,
                num_proc=2, desc="Tokenisation")

    def grouper(lot):
        tous = list(itertools.chain.from_iterable(lot["input_ids"]))
        n = (len(tous) // longueur) * longueur
        return {"input_ids": [tous[i:i + longueur] for i in range(0, n, longueur)]}

    ds = ds.map(grouper, batched=True, remove_columns=ds.column_names,
                num_proc=2, desc="Découpage en blocs")
    return ds


def preparer_sft(ds, tok, longueur):
    """
    Mode "sft" (phase D : apprendre à répondre).

    Chaque exemple devient :
        <|user|> Conjugue gukora au futur. <|assistant|> Nzokora, uzokora...<|endoftext|>

    IMPORTANT : on met -100 sur la partie question. Ça veut dire
    "ne compte pas d'erreur ici". Le modèle apprend donc à produire
    la RÉPONSE, pas à recopier la question.

    Deux formats de lignes .jsonl sont acceptés :
      - 1 tour  : {"user": "...", "assistant": "..."}
      - plusieurs tours : {"messages": [{"role": "user", "content": "..."},
                                        {"role": "assistant", "content": "..."}, ...]}
    Plusieurs tours donnent :
        <|user|> Bwakeye <|assistant|> Bwakeye neza.<|endoftext|><|user|> N'amaki? <|assistant|> N'amahoro.<|endoftext|>
    Le modèle apprend toutes les réponses, et à tenir compte de ce qui a été dit avant.
    """
    def construire(ex):
        messages = ex.get("messages") or [
            {"role": "user", "content": ex["user"]},
            {"role": "assistant", "content": ex["assistant"]},
        ]
        ids, labels = [], []
        for m in messages:
            if m["role"] == "user":
                morceau = tok(f"<|user|> {m['content']} <|assistant|>")["input_ids"]
                labels += [-100] * len(morceau)
            else:
                morceau = tok(" " + m["content"] + tok.eos_token)["input_ids"]
                labels += morceau
            ids += morceau
        if len(ids) > longueur:
            # Trop long : on coupe, mais on garde <|endoftext|> à la fin pour que le modèle
            # apprenne toujours à s'arrêter (avant : la fin était perdue).
            ids = ids[:longueur - 1] + [tok.eos_token_id]
            labels = labels[:longueur - 1] + [tok.eos_token_id]
        return {"input_ids": ids, "labels": labels}

    return ds.map(construire, remove_columns=ds.column_names, desc="Préparation SFT")


def faire_collate(tok, mode):
    """
    Le "collate" assemble plusieurs exemples en un paquet (batch) pour le GPU.
    """
    pad = tok.pad_token_id

    def collate(exemples):
        if mode == "pretrain":
            ids = torch.tensor([e["input_ids"] for e in exemples])
            # MÉCANISME 4 : labels = le texte lui-même.
            # Le modèle décale tout seul d'une position : à chaque place,
            # il doit deviner le morceau SUIVANT.
            return {"input_ids": ids,
                    "attention_mask": torch.ones_like(ids),
                    "labels": ids.clone()}

        # Mode sft : les exemples n'ont pas la même longueur, on complète.
        m = max(len(e["input_ids"]) for e in exemples)
        ids, labels, masque = [], [], []
        for e in exemples:
            manque = m - len(e["input_ids"])
            ids.append(e["input_ids"] + [pad] * manque)
            labels.append(e["labels"] + [-100] * manque)
            masque.append([1] * len(e["input_ids"]) + [0] * manque)
        return {"input_ids": torch.tensor(ids),
                "attention_mask": torch.tensor(masque),
                "labels": torch.tensor(labels)}

    return collate


# ---------------------------------------------------------------------------
# 3. LE MODÈLE
# ---------------------------------------------------------------------------

def creer_modele_vide(architecture, tok):
    """
    Modèle "depuis zéro" : tous les boutons (paramètres) sont tirés au hasard.
    Il ne sait RIEN au départ, ni anglais, ni français, ni kirundi.
    On garde la forme GPT-NeoX (comme Pythia) car le backend sait déjà la lire.

    "architecture" vient du YAML, par exemple :
        n_couches: 12, largeur: 768, tetes: 12, contexte: 1024
    """
    from transformers import GPTNeoXConfig

    config = GPTNeoXConfig(
        vocab_size=len(tok),
        hidden_size=architecture["largeur"],
        num_hidden_layers=architecture["n_couches"],
        num_attention_heads=architecture["tetes"],
        intermediate_size=4 * architecture["largeur"],
        max_position_embeddings=architecture["contexte"],
        rotary_pct=0.25,
        use_parallel_residual=True,
        tie_word_embeddings=False,
        bos_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    return AutoModelForCausalLM.from_config(config)


def preparer_modele(chemin, tok, reinitialiser_embeddings, architecture=None):
    """
    Charge le modèle :
      - "architecture" donnée et pas de reprise : modèle VIDE (départ de zéro),
      - sinon : une sauvegarde (phase précédente ou reprise après coupure).
    """
    if architecture is not None:
        model = creer_modele_vide(architecture, tok)
    else:
        try:  # transformers récent : "dtype" ; ancien : "torch_dtype"
            model = AutoModelForCausalLM.from_pretrained(chemin, dtype=torch.float32)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(chemin, torch_dtype=torch.float32)

    if reinitialiser_embeddings:
        # MÉCANISME 2 : LES EMBEDDINGS.
        # Pythia-14m a une table de 50 304 morceaux ANGLAIS. Notre tokenizer
        # a ses propres morceaux français + kirundi. Les anciennes lignes ne
        # veulent plus rien dire, donc on change la taille de la table et on
        # la remplit de petits nombres au hasard. L'entraînement placera
        # ensuite chaque morceau au bon endroit ("umwana" près de "enfant").
        #
        # Les couches d'attention (le "cerveau") sont GARDÉES.
        log(f"Nouvelle table de mots : {model.config.vocab_size} -> {len(tok)}")
        model.resize_token_embeddings(len(tok))
        ecart = model.config.initializer_range  # 0.02 en général
        torch.nn.init.normal_(model.get_input_embeddings().weight, 0.0, ecart)
        torch.nn.init.normal_(model.get_output_embeddings().weight, 0.0, ecart)

    # Le modèle doit connaître les numéros de nos mots spéciaux.
    model.config.eos_token_id = tok.eos_token_id
    model.config.bos_token_id = tok.eos_token_id
    model.config.pad_token_id = tok.pad_token_id
    if model.generation_config is not None:
        model.generation_config.eos_token_id = tok.eos_token_id
        model.generation_config.pad_token_id = tok.pad_token_id

    n = sum(p.numel() for p in model.parameters())
    log(f"Modèle chargé : {n / 1e6:.1f} millions de paramètres")
    return model


def creer_optimiseur(model, lr, poids_decay):
    """
    MÉCANISME 5 : L'OPTIMISEUR (AdamW).
    C'est lui qui tourne les "boutons" du modèle après chaque erreur.

    Le "weight decay" pousse doucement les boutons vers zéro, pour éviter
    que le modèle apprenne par cœur. On ne l'applique pas aux biais ni aux
    normalisations (LayerNorm), c'est la pratique habituelle.
    """
    avec, sans = [], []
    for nom, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (sans if p.ndim < 2 or "norm" in nom.lower() else avec).append(p)
    groupes = [{"params": avec, "weight_decay": poids_decay},
               {"params": sans, "weight_decay": 0.0}]
    return torch.optim.AdamW(groupes, lr=lr, betas=(0.9, 0.95), eps=1e-8)


# ---------------------------------------------------------------------------
# 4. ÉVALUATION : est-ce que le modèle progresse ?
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluer(model, loader, appareil, type_calcul, max_lots=50):
    """
    On mesure l'erreur (loss) sur des textes que le modèle n'a JAMAIS vus
    pendant l'entraînement. Si elle baisse, il apprend vraiment.
    Si elle monte alors que la loss d'entraînement baisse, il apprend par
    cœur (on appelle ça le surapprentissage).

    Perplexité = exp(loss). En gros : "entre combien de mots le modèle
    hésite". 500 = il devine presque au hasard. 20 = il est bon.
    """
    model.eval()
    total, n = 0.0, 0
    for i, lot in enumerate(loader):
        if i >= max_lots:
            break
        lot = {k: v.to(appareil) for k, v in lot.items()}
        with torch.autocast("cuda", dtype=type_calcul, enabled=str(appareil).startswith("cuda")):
            total += model(**lot).loss.item()
        n += 1
    model.train()
    loss = total / max(n, 1)
    return loss, math.exp(min(loss, 20))


@torch.no_grad()
def montrer_exemples(model, tok, phrases, appareil, penalite=1.0, sans_repeter=0):
    """
    On demande au modèle de continuer quelques phrases.
    C'est la façon la plus simple de VOIR ses progrès en orthographe
    et en grammaire.
    """
    model.eval()
    for debut in phrases:
        ids = tok(debut, return_tensors="pt", return_token_type_ids=False).to(appareil)
        sortie = model.generate(**ids, max_new_tokens=30, do_sample=True,
                                temperature=0.7, top_p=0.9, repetition_penalty=penalite,
                                no_repeat_ngram_size=sans_repeter, pad_token_id=tok.pad_token_id)
        log(f"  >> {tok.decode(sortie[0], skip_special_tokens=True)!r}")
    model.train()


# ---------------------------------------------------------------------------
# 5. SAUVEGARDE ET REPRISE (Colab coupe souvent !)
# ---------------------------------------------------------------------------

def sauvegarder(model, tok, optimiseur, planning, scaler, etat, dossier):
    """
    On sauvegarde sur Google Drive :
      - le modèle et le tokenizer (pour l'utiliser plus tard),
      - l'état de l'entraînement (pour reprendre si Colab s'arrête).
    """
    os.makedirs(dossier, exist_ok=True)
    model.save_pretrained(dossier)
    tok.save_pretrained(dossier)
    torch.save({"optimiseur": optimiseur.state_dict(),
                "planning": planning.state_dict(),
                "scaler": scaler.state_dict(),
                **etat}, os.path.join(dossier, "etat_entrainement.pt"))
    log(f"Sauvegardé dans {dossier}")


# ---------------------------------------------------------------------------
# 6. LA BOUCLE D'ENTRAÎNEMENT (le cœur de tout)
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    cfg = charger_config(parser.parse_args().config)

    local = init_distribue()
    fixer_hasard(cfg.get("graine", 42))
    appareil = "cuda" if torch.cuda.is_available() else "cpu"
    if appareil == "cuda" and local is not None:
        appareil = f"cuda:{local}"
    sur_gpu = appareil.startswith("cuda")
    debut_session = time.time()
    fin_session = float(os.environ.get("FIN_SESSION", "0")) or None
    marge = cfg.get("marge_minutes", 25) * 60
    if appareil == "cpu":
        log("ATTENTION : pas de GPU. Dans Colab : Exécution > Modifier le type d'exécution > GPU")

    # bfloat16 sur les GPU récents (A100, L4), float16 sur T4, T600 et P100.
    # Ces formats "courts" vont 2 fois plus vite et prennent moins de mémoire.
    # On teste la génération du GPU (8 = Ampere) : torch.cuda.is_bf16_supported()
    # répond "oui" même sur T4 / T600 (bfloat16 émulé, donc LENT).
    if sur_gpu and torch.cuda.get_device_capability()[0] >= 8:
        type_calcul = torch.bfloat16
    else:
        type_calcul = torch.float16

    sortie = cfg["dossier_sortie"]
    dossier_reprise = os.path.join(sortie, "derniere")
    fichier_etat = os.path.join(dossier_reprise, "etat_entrainement.pt")
    reprise = os.path.exists(fichier_etat)
    marque_pause = os.path.join(sortie, "PAUSE")
    if RANG == 0 and os.path.exists(marque_pause):
        os.remove(marque_pause)   # on reprend : la pause précédente est finie

    # --- Tokenizer et modèle -------------------------------------------------
    # Si une sauvegarde existe, on repart d'elle (Colab a coupé).
    tok = charger_tokenizer(dossier_reprise if reprise else cfg["tokenizer"])
    model = preparer_modele(
        dossier_reprise if reprise else cfg.get("modele"),
        tok,
        reinitialiser_embeddings=(not reprise and cfg.get("reinitialiser_embeddings", False)),
        architecture=None if reprise else cfg.get("architecture"),
    ).to(appareil)

    # Économise la mémoire GPU (utile pour le modèle de 78M sur un T4).
    if cfg.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
    # "dropout" : coupe au hasard une part des signaux pendant l'entraînement (contre le par-cœur)
    if cfg.get("dropout") is not None:
        p_drop = float(cfg["dropout"])
        for attr in ("attention_dropout", "hidden_dropout", "resid_pdrop", "attn_pdrop", "embd_pdrop"):
            if hasattr(model.config, attr):
                setattr(model.config, attr, p_drop)
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = p_drop
        log(f"Dropout {p_drop}")
    model.train()
    modele_ddp = model
    if NB_GPU > 1:
        from torch.nn.parallel import DistributedDataParallel
        modele_ddp = DistributedDataParallel(model, device_ids=[local] if sur_gpu else None)

    # --- Données -------------------------------------------------------------
    # Plusieurs GPU : le premier prépare (et met en cache), les autres relisent le cache.
    mode = cfg.get("mode", "pretrain")
    longueur = cfg.get("longueur", 512)
    for tour in (0, 1):
        if (RANG == 0) == (tour == 0):
            preparer = preparer_pretrain if mode == "pretrain" else preparer_sft
            format_fichier = "text" if mode == "pretrain" else "json"
            if cfg.get("eval_par_source"):
                # Évaluation mise de côté AVANT les répétitions (pas de copies dans les deux)
                brut, brut_eval = charger_sources(cfg["sources"], format_fichier,
                                                  part_eval=cfg.get("part_eval", 0.01))
                coupe = {"train": preparer(brut, tok, longueur), "test": preparer(brut_eval, tok, longueur)}
            else:
                brut = charger_sources(cfg["sources"], format_fichier)
                ds = preparer(brut, tok, longueur)
                # On met de côté 1 % des données pour l'évaluation.
                coupe = ds.train_test_split(test_size=cfg.get("part_eval", 0.01), seed=42)
        attendre_tous()
    ds_train, ds_eval = coupe["train"], coupe["test"]
    log(f"Entraînement : {len(ds_train)} exemples | Évaluation : {len(ds_eval)}")

    taille_lot = cfg.get("taille_lot", 16)
    # Plusieurs GPU : chacun accumule moins, le pas reste aussi gros qu'avec un seul GPU.
    accumulation = max(1, cfg.get("accumulation", 4) // NB_GPU)
    if NB_GPU > 1:
        log(f"{NB_GPU} GPU : accumulation {accumulation} par GPU")
    collate = faire_collate(tok, mode)
    loader_eval = DataLoader(ds_eval, batch_size=taille_lot, collate_fn=collate)

    # --- Optimiseur et planning du learning rate ------------------------------
    epoques = cfg.get("epoques", 1)
    from torch.utils.data.distributed import DistributedSampler
    echantillonneur = (DistributedSampler(ds_train, num_replicas=NB_GPU, rank=RANG, shuffle=True,
                                          seed=cfg.get("graine", 42)) if NB_GPU > 1 else None)
    lots_par_epoque = math.ceil((len(echantillonneur) if echantillonneur else len(ds_train)) / taille_lot)
    pas_par_epoque = math.ceil(lots_par_epoque / accumulation)
    # "min_pas" : assez d'époques pour faire au moins ce nombre de pas (lecture plus longue)
    if cfg.get("min_pas") and epoques * pas_par_epoque < cfg["min_pas"]:
        epoques = math.ceil(cfg["min_pas"] / pas_par_epoque)
        log(f"min_pas {cfg['min_pas']} : {epoques} époques")
    pas_total = epoques * pas_par_epoque

    optimiseur = creer_optimiseur(model, cfg.get("learning_rate", 6e-4),
                                  cfg.get("weight_decay", 0.1))

    # Le learning rate (taille du pas) :
    #   - monte doucement au début (warmup) pour ne pas "casser" le modèle,
    #     surtout juste après un agrandissement avec grow.py,
    #   - puis descend en douceur (cosinus) pour affiner à la fin.
    planning = get_cosine_schedule_with_warmup(
        optimiseur, cfg.get("warmup", 1000), pas_total)

    # Le GradScaler évite que les très petits nombres deviennent 0 en float16.
    scaler = torch.amp.GradScaler("cuda", enabled=(type_calcul == torch.float16 and sur_gpu))

    pas = 0
    if reprise:
        etat = torch.load(fichier_etat, map_location=appareil, weights_only=False)
        optimiseur.load_state_dict(etat["optimiseur"])
        planning.load_state_dict(etat["planning"])
        scaler.load_state_dict(etat["scaler"])
        pas = etat["pas"]
        log(f"Reprise au pas {pas} / {pas_total}")

    # Où en étions-nous ? (en nombre de lots déjà vus)
    lots_deja_vus = pas * accumulation
    epoque_depart = lots_deja_vus // lots_par_epoque

    log(f"Début : {pas_total} pas au total, {epoques} époque(s)")
    fichier_meilleure = os.path.join(sortie, "meilleure", "loss_eval.txt")
    meilleure_loss_eval = (float(open(fichier_meilleure).read()) if os.path.exists(fichier_meilleure)
                           else float("inf"))
    loss_cumulee, debut = 0.0, time.time()
    pause = False
    heures = cfg.get("sauver_toutes_les_heures")
    prochaine_copie = debut_session + heures * 3600 if heures else None

    def doit_s_arreter():
        """Plus assez de temps avant la fin de la session ? (même décision sur tous les GPU)"""
        stop = fin_session is not None and time.time() > fin_session - marge
        if NB_GPU > 1:
            t = torch.tensor([int(stop)], device=appareil if sur_gpu else "cpu")
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX)
            stop = bool(t.item())
        return stop

    for epoque in range(epoque_depart, epoques):
        if cfg.get("max_pas") and pas >= cfg["max_pas"]:
            break
        # On recrée le loader à chaque époque avec une graine fixe :
        # en cas de reprise, l'ordre des lots est exactement le même.
        if echantillonneur:
            echantillonneur.set_epoch(epoque)
            loader = DataLoader(ds_train, batch_size=taille_lot, sampler=echantillonneur,
                                collate_fn=collate, num_workers=2, pin_memory=sur_gpu)
        else:
            g = torch.Generator().manual_seed(cfg.get("graine", 42) + epoque)
            loader = DataLoader(ds_train, batch_size=taille_lot, shuffle=True,
                                generator=g, collate_fn=collate, num_workers=2,
                                pin_memory=sur_gpu)
        a_sauter = lots_deja_vus - epoque * lots_par_epoque

        for i, lot in enumerate(loader):
            if i < a_sauter:
                continue  # déjà vu avant la coupure de Colab
            lot = {k: v.to(appareil, non_blocking=True) for k, v in lot.items()}

            # ---- MÉCANISMES 2, 3, 4 : le modèle LIT et DEVINE -----------------
            # À l'intérieur de model(...) :
            #   numéros -> embeddings (sens des mots)
            #   -> attention x N couches (liens entre les mots :
            #      "Abana" <-> "ba-ragiye", "filles" <-> "parties")
            #   -> prédiction du morceau suivant à chaque position
            #   -> loss : à quel point il s'est trompé
            dernier_du_pas = (i + 1) % accumulation == 0
            # Plusieurs GPU : ils ne mettent leurs corrections en commun qu'au dernier lot du pas.
            from contextlib import nullcontext
            synchro = nullcontext() if (dernier_du_pas or NB_GPU == 1) else modele_ddp.no_sync()
            with synchro:
                with torch.autocast("cuda", dtype=type_calcul, enabled=sur_gpu):
                    loss = modele_ddp(**lot).loss

                # On divise par "accumulation" : on additionne les erreurs de
                # plusieurs petits lots pour simuler un gros lot (mémoire GPU limitée).
                loss_cumulee += loss.item()
                loss = loss / accumulation

                # ---- MÉCANISME 5 : BACKPROPAGATION ------------------------------
                # Calcule, pour chacun des millions de boutons, dans quel sens
                # le tourner pour faire moins d'erreur.
                scaler.scale(loss).backward()

            if not dernier_du_pas:
                continue

            # On limite la force des corrections : une phrase très bizarre
            # ne doit pas tout casser d'un coup.
            scaler.unscale_(optimiseur)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # ---- MÉCANISME 5 (suite) : on TOURNE les boutons ------------------
            scaler.step(optimiseur)
            scaler.update()
            optimiseur.zero_grad(set_to_none=True)
            planning.step()
            pas += 1
            if cfg.get("max_pas") and pas >= cfg["max_pas"]:
                break  # essai rapide : on s'arrête là

            # ---- Suivi ---------------------------------------------------------
            if pas % cfg.get("log_tous_les", 50) == 0:
                moy = loss_cumulee / (cfg.get("log_tous_les", 50) * accumulation)
                vitesse = (time.time() - debut) / cfg.get("log_tous_les", 50)
                log(f"époque {epoque + 1} | pas {pas}/{pas_total} | loss {moy:.3f} | "
                    f"lr {planning.get_last_lr()[0]:.2e} | {vitesse:.2f} s/pas")
                loss_cumulee, debut = 0.0, time.time()

            if pas % cfg.get("eval_tous_les", 500) == 0:
                if RANG == 0:
                    loss_eval, ppl = evaluer(model, loader_eval, appareil, type_calcul, cfg.get("lots_eval", 50))
                    log(f"ÉVALUATION | loss {loss_eval:.3f} | perplexité {ppl:.1f}")
                    montrer_exemples(model, tok, cfg.get("phrases_test", []), appareil,
                                     cfg.get("penalite_repetition", 1.0), cfg.get("sans_repeter_ngram", 0))
                    if loss_eval < meilleure_loss_eval:
                        meilleure_loss_eval = loss_eval
                        model.save_pretrained(os.path.join(sortie, "meilleure"))
                        tok.save_pretrained(os.path.join(sortie, "meilleure"))
                        open(fichier_meilleure, "w").write(str(loss_eval))
                        log("Nouveau meilleur modèle gardé dans 'meilleure'")
                attendre_tous()

            if pas % cfg.get("sauver_tous_les", 1000) == 0 and RANG == 0:
                sauvegarder(model, tok, optimiseur, planning, scaler,
                            {"pas": pas}, dossier_reprise)

            # Copie de sécurité toutes les N heures (en plus de "derniere")
            if prochaine_copie and time.time() >= prochaine_copie:
                h = round((time.time() - debut_session) / 3600)
                if RANG == 0:
                    sauvegarder(model, tok, optimiseur, planning, scaler, {"pas": pas}, dossier_reprise)
                    model.save_pretrained(os.path.join(sortie, f"sauvegarde_{h}h"))
                    tok.save_pretrained(os.path.join(sortie, f"sauvegarde_{h}h"))
                    log(f"Copie des {h} h gardée dans 'sauvegarde_{h}h' (pas {pas})")
                prochaine_copie += heures * 3600
                attendre_tous()

            if doit_s_arreter():
                pause = True
                break
        if pause:
            break

    if pause:
        if RANG == 0:
            sauvegarder(model, tok, optimiseur, planning, scaler, {"pas": pas}, dossier_reprise)
            open(marque_pause, "w").write(str(pas))
            log(f"PAUSE au pas {pas}/{pas_total} : la session finit bientôt. "
                "La version suivante du notebook reprendra ici.")
        attendre_tous()
        if NB_GPU > 1:
            torch.distributed.destroy_process_group()
        return

    # --- Fin ---------------------------------------------------------------------
    if RANG == 0:
        # Même mesure que pendant l'entraînement (avant : 200 lots contre 50 -> comparaison fausse)
        loss_eval, ppl = evaluer(model, loader_eval, appareil, type_calcul, cfg.get("lots_eval", 50))
        log(f"FIN | loss évaluation {loss_eval:.3f} | perplexité {ppl:.1f}")
        montrer_exemples(model, tok, cfg.get("phrases_test", []), appareil,
                         cfg.get("penalite_repetition", 1.0), cfg.get("sans_repeter_ngram", 0))
        finale = os.path.join(sortie, "finale")
        if cfg.get("garder_meilleure") and loss_eval > meilleure_loss_eval:
            # "garder_meilleure" : la fin a moins bien fait que le meilleur point -> on garde le meilleur
            import shutil
            shutil.copytree(os.path.join(sortie, "meilleure"), finale, dirs_exist_ok=True)
            log(f"finale = meilleure (loss {meilleure_loss_eval:.3f} < {loss_eval:.3f} à la fin)")
        else:
            model.save_pretrained(finale)
            tok.save_pretrained(finale)
        log(f"Modèle final dans {finale} : c'est lui qu'on donne à la phase suivante.")
    attendre_tous()
    if NB_GPU > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
