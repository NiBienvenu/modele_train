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
    """Lit le fichier YAML de la phase (phaseA.yaml, phaseB.yaml, ...)."""
    with open(chemin, encoding="utf-8") as f:
        return yaml.safe_load(f)


def fixer_hasard(graine):
    """Même graine = mêmes résultats. Utile pour comparer deux essais."""
    random.seed(graine)
    torch.manual_seed(graine)
    torch.cuda.manual_seed_all(graine)


def log(msg):
    print(time.strftime("[%H:%M:%S] "), msg, flush=True)


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

def charger_sources(sources, format_fichier, decoupage="paragraph"):
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
    morceaux = []
    for src in sources:
        if format_fichier == "text":
            # sample_by="paragraph" : un exemple = un paragraphe entier
            # (séparé par une ligne vide). Le modèle garde ainsi le contexte.
            ds = load_dataset("text", data_files=src["fichiers"],
                              sample_by=decoupage, split="train")
            ds = ds.filter(lambda ex: len(ex["text"].strip()) > 0)
        else:
            # Pour la phase D : fichiers .jsonl avec {"user": ..., "assistant": ...}
            ds = load_dataset("json", data_files=src["fichiers"], split="train")

        n = int(src.get("repeter", 1))
        log(f"Source {src['fichiers']} : {len(ds)} exemples x {n}")
        morceaux.extend([ds] * n)

    # On mélange tout pour que le modèle ne voie pas une seule langue
    # pendant des heures, puis l'autre.
    return concatenate_datasets(morceaux).shuffle(seed=42)


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
    """
    def construire(ex):
        question = tok(f"<|user|> {ex['user']} <|assistant|>")["input_ids"]
        reponse = tok(" " + ex["assistant"] + tok.eos_token)["input_ids"]
        ids = (question + reponse)[:longueur]
        labels = ([-100] * len(question) + reponse)[:longueur]
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

def preparer_modele(chemin, tok, reinitialiser_embeddings):
    """
    Charge le modèle : Pythia-14m au tout début, puis nos propres
    sauvegardes ensuite (ou le modèle agrandi par grow.py).
    """
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
        with torch.autocast("cuda", dtype=type_calcul, enabled=appareil == "cuda"):
            total += model(**lot).loss.item()
        n += 1
    model.train()
    loss = total / max(n, 1)
    return loss, math.exp(min(loss, 20))


@torch.no_grad()
def montrer_exemples(model, tok, phrases, appareil):
    """
    On demande au modèle de continuer quelques phrases.
    C'est la façon la plus simple de VOIR ses progrès en orthographe
    et en grammaire.
    """
    model.eval()
    for debut in phrases:
        ids = tok(debut, return_tensors="pt").to(appareil)
        sortie = model.generate(**ids, max_new_tokens=30, do_sample=True,
                                temperature=0.7, top_p=0.9,
                                pad_token_id=tok.pad_token_id)
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

    fixer_hasard(cfg.get("graine", 42))
    appareil = "cuda" if torch.cuda.is_available() else "cpu"
    if appareil == "cpu":
        log("ATTENTION : pas de GPU. Dans Colab : Exécution > Modifier le type d'exécution > GPU")

    # bfloat16 sur les GPU récents (A100, L4), float16 sur T4 et P100.
    # Ces formats "courts" vont 2 fois plus vite et prennent moins de mémoire.
    if appareil == "cuda" and torch.cuda.is_bf16_supported():
        type_calcul = torch.bfloat16
    else:
        type_calcul = torch.float16

    sortie = cfg["dossier_sortie"]
    dossier_reprise = os.path.join(sortie, "derniere")
    fichier_etat = os.path.join(dossier_reprise, "etat_entrainement.pt")
    reprise = os.path.exists(fichier_etat)

    # --- Tokenizer et modèle -------------------------------------------------
    # Si une sauvegarde existe, on repart d'elle (Colab a coupé).
    tok = charger_tokenizer(dossier_reprise if reprise else cfg["tokenizer"])
    model = preparer_modele(
        dossier_reprise if reprise else cfg["modele"],
        tok,
        reinitialiser_embeddings=(not reprise and cfg.get("reinitialiser_embeddings", False)),
    ).to(appareil)

    # Économise la mémoire GPU (utile pour le modèle de 78M sur un T4).
    if cfg.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
    model.train()

    # --- Données -------------------------------------------------------------
    mode = cfg.get("mode", "pretrain")
    longueur = cfg.get("longueur", 512)
    brut = charger_sources(cfg["sources"], "text" if mode == "pretrain" else "json")
    ds = (preparer_pretrain(brut, tok, longueur) if mode == "pretrain"
          else preparer_sft(brut, tok, longueur))

    # On met de côté 1 % des données pour l'évaluation.
    coupe = ds.train_test_split(test_size=cfg.get("part_eval", 0.01), seed=42)
    ds_train, ds_eval = coupe["train"], coupe["test"]
    log(f"Entraînement : {len(ds_train)} exemples | Évaluation : {len(ds_eval)}")

    taille_lot = cfg.get("taille_lot", 16)
    accumulation = cfg.get("accumulation", 4)
    collate = faire_collate(tok, mode)
    loader_eval = DataLoader(ds_eval, batch_size=taille_lot, collate_fn=collate)

    # --- Optimiseur et planning du learning rate ------------------------------
    epoques = cfg.get("epoques", 1)
    lots_par_epoque = math.ceil(len(ds_train) / taille_lot)
    pas_total = epoques * math.ceil(lots_par_epoque / accumulation)

    optimiseur = creer_optimiseur(model, cfg.get("learning_rate", 6e-4),
                                  cfg.get("weight_decay", 0.1))

    # Le learning rate (taille du pas) :
    #   - monte doucement au début (warmup) pour ne pas "casser" le modèle,
    #     surtout juste après un agrandissement avec grow.py,
    #   - puis descend en douceur (cosinus) pour affiner à la fin.
    planning = get_cosine_schedule_with_warmup(
        optimiseur, cfg.get("warmup", 1000), pas_total)

    # Le GradScaler évite que les très petits nombres deviennent 0 en float16.
    scaler = torch.amp.GradScaler("cuda", enabled=(type_calcul == torch.float16
                                                   and appareil == "cuda"))

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
    meilleure_loss_eval = float("inf")
    loss_cumulee, debut = 0.0, time.time()

    for epoque in range(epoque_depart, epoques):
        # On recrée le loader à chaque époque avec une graine fixe :
        # en cas de reprise, l'ordre des lots est exactement le même.
        g = torch.Generator().manual_seed(cfg.get("graine", 42) + epoque)
        loader = DataLoader(ds_train, batch_size=taille_lot, shuffle=True,
                            generator=g, collate_fn=collate, num_workers=2,
                            pin_memory=(appareil == "cuda"))
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
            with torch.autocast("cuda", dtype=type_calcul, enabled=appareil == "cuda"):
                loss = model(**lot).loss

            # On divise par "accumulation" : on additionne les erreurs de
            # plusieurs petits lots pour simuler un gros lot (mémoire GPU limitée).
            loss_cumulee += loss.item()
            loss = loss / accumulation

            # ---- MÉCANISME 5 : BACKPROPAGATION ----------------------------------
            # Calcule, pour chacun des millions de boutons, dans quel sens
            # le tourner pour faire moins d'erreur.
            scaler.scale(loss).backward()

            if (i + 1) % accumulation != 0:
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

            # ---- Suivi ---------------------------------------------------------
            if pas % cfg.get("log_tous_les", 50) == 0:
                moy = loss_cumulee / (cfg.get("log_tous_les", 50) * accumulation)
                vitesse = (time.time() - debut) / cfg.get("log_tous_les", 50)
                log(f"époque {epoque + 1} | pas {pas}/{pas_total} | loss {moy:.3f} | "
                    f"lr {planning.get_last_lr()[0]:.2e} | {vitesse:.2f} s/pas")
                loss_cumulee, debut = 0.0, time.time()

            if pas % cfg.get("eval_tous_les", 500) == 0:
                loss_eval, ppl = evaluer(model, loader_eval, appareil, type_calcul)
                log(f"ÉVALUATION | loss {loss_eval:.3f} | perplexité {ppl:.1f}")
                montrer_exemples(model, tok, cfg.get("phrases_test", []), appareil)
                if loss_eval < meilleure_loss_eval:
                    meilleure_loss_eval = loss_eval
                    model.save_pretrained(os.path.join(sortie, "meilleure"))
                    tok.save_pretrained(os.path.join(sortie, "meilleure"))
                    log("Nouveau meilleur modèle gardé dans 'meilleure'")

            if pas % cfg.get("sauver_tous_les", 1000) == 0:
                sauvegarder(model, tok, optimiseur, planning, scaler,
                            {"pas": pas}, dossier_reprise)

    # --- Fin ---------------------------------------------------------------------
    loss_eval, ppl = evaluer(model, loader_eval, appareil, type_calcul, max_lots=200)
    log(f"FIN | loss évaluation {loss_eval:.3f} | perplexité {ppl:.1f}")
    montrer_exemples(model, tok, cfg.get("phrases_test", []), appareil)
    finale = os.path.join(sortie, "finale")
    model.save_pretrained(finale)
    tok.save_pretrained(finale)
    log(f"Modèle final dans {finale} : c'est lui qu'on donne à la phase suivante.")


if __name__ == "__main__":
    main()
