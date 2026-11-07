#!/usr/bin/env python
"""
grow.py : fait GRANDIR un modèle GPT-NeoX (Pythia) en gardant ce qu'il sait.

    Pythia-14m       : largeur 128, 6 couches, 4 têtes   (~14M avec le vocabulaire anglais)
    Après phase A    : même taille, mais NOTRE vocabulaire fr+rn (24 000 morceaux)
    Cible conseillée : largeur 768, 12 couches, 12 têtes (~122M)

COMMENT ON GARDE LE SAVOIR
  1) LARGEUR : chaque matrice du petit modèle est copiée dans le coin
     de la grande matrice ; le reste reçoit de tout petits nombres. Le
     grand modèle commence donc presque comme le petit.
  2) ATTENTION : les poids q, k, v sont rangés TÊTE PAR TÊTE. On copie
     la tête j du petit modèle dans la tête j du grand (et pas en vrac,
     sinon on mélangerait les têtes).
  3) PROFONDEUR : chaque couche du petit modèle est dupliquée :
     6 couches -> 12 couches = chaque couche copiée 2 fois.
  4) NORMALISATIONS : copiées ; les nouvelles dimensions valent 1 (poids)
     et 0 (biais), la valeur neutre.

Après un agrandissement, il faut un "warmup" doux (1000 à 2000 pas) :
le modèle doit s'habituer à sa nouvelle taille.

Utilisation :
    python grow.py --source /content/drive/MyDrive/modele/phaseA/finale \
                   --sortie /content/drive/MyDrive/modele/grand_init \
                   --hidden 768 --layers 12 --heads 12
"""

import argparse
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTNeoXConfig, GPTNeoXForCausalLM


def compter(model):
    return sum(p.numel() for p in model.parameters())


def copier_coin(petit, grand):
    """Copie `petit` dans le coin haut-gauche de `grand` (même nombre de dimensions)."""
    idx = tuple(slice(0, min(a, b)) for a, b in zip(petit.shape, grand.shape))
    grand[idx] = petit[idx]


def copier_qkv(petit, grand, h_petit, h_grand, n_petit, n_grand, est_biais):
    """
    GPT-NeoX range query_key_value par tête : [têtes, 3 (q,k,v), dim_tête].
    On copie la tête j (q, k et v) du petit modèle dans la tête j du grand.
    """
    d_petit, d_grand = h_petit // n_petit, h_grand // n_grand
    if est_biais:
        p = petit.view(n_petit, 3, d_petit)
        g = grand.view(n_grand, 3, d_grand)
        n, d = min(n_petit, n_grand), min(d_petit, d_grand)
        g[:n, :, :d] = p[:n, :, :d]
    else:
        p = petit.view(n_petit, 3, d_petit, h_petit)
        g = grand.view(n_grand, 3, d_grand, h_grand)
        n, d, h = min(n_petit, n_grand), min(d_petit, d_grand), min(h_petit, h_grand)
        g[:n, :, :d, :h] = p[:n, :, :d, :h]


@torch.no_grad()
def agrandir(petit, hidden, layers, heads, inter=None):
    cp = petit.config
    # On copie TOUTE la configuration du petit modèle (rotary, normes...)
    # et on change seulement la taille : marche avec toutes les versions
    # de transformers.
    reglages = cp.to_dict()
    reglages.update(hidden_size=hidden, num_hidden_layers=layers,
                    num_attention_heads=heads, intermediate_size=inter or 4 * hidden,
                    tie_word_embeddings=False)
    for cle in ("architectures", "_name_or_path", "transformers_version", "model_type"):
        reglages.pop(cle, None)
    cfg = GPTNeoXConfig(**reglages)
    grand = GPTNeoXForCausalLM(cfg)
    sd_p, sd_g = petit.state_dict(), grand.state_dict()

    # Petits nombres partout (le "bruit" des nouvelles parties), neutres pour les normes
    for nom, w in sd_g.items():
        if "layernorm" in nom or "layer_norm" in nom:
            w.fill_(1.0 if nom.endswith("weight") else 0.0)
        elif nom.endswith("bias"):
            w.zero_()
        else:
            w.normal_(0.0, 0.02 / (2 * layers) ** 0.5 if "dense.weight" in nom or "4h_to_h" in nom else 0.02)

    n_petit = cp.num_hidden_layers
    for nom, w in sd_g.items():
        source = nom
        m = re.search(r"\.layers\.(\d+)\.", nom)
        if m:
            i = int(m.group(1))
            j = i * n_petit // layers            # duplication des couches
            source = nom.replace(f".layers.{i}.", f".layers.{j}.")
        if source not in sd_p:
            continue
        wp = sd_p[source]
        if "query_key_value" in nom:
            copier_qkv(wp, w, cp.hidden_size, hidden, cp.num_attention_heads, heads,
                       est_biais=nom.endswith("bias"))
        elif wp.dim() == w.dim():
            copier_coin(wp, w)
    grand.load_state_dict(sd_g)
    return grand


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--sortie", required=True)
    p.add_argument("--hidden", type=int, default=768)
    p.add_argument("--layers", type=int, default=12)
    p.add_argument("--heads", type=int, default=12)
    p.add_argument("--inter", type=int, default=None)
    args = p.parse_args()

    petit = AutoModelForCausalLM.from_pretrained(args.source, dtype=torch.float32)
    print(f"Modèle de départ : {compter(petit) / 1e6:.1f}M paramètres "
          f"(largeur {petit.config.hidden_size}, {petit.config.num_hidden_layers} couches)")
    grand = agrandir(petit, args.hidden, args.layers, args.heads, args.inter)
    print(f"Modèle agrandi   : {compter(grand) / 1e6:.1f}M paramètres "
          f"(largeur {args.hidden}, {args.layers} couches, {args.heads} têtes)")
    grand.save_pretrained(args.sortie)
    try:
        AutoTokenizer.from_pretrained(args.source).save_pretrained(args.sortie)
    except Exception:
        print("(pas de tokenizer à copier dans la source)")
    print(f"Sauvegardé dans {args.sortie}. Lance ensuite train.py avec un warmup de 1000 à 2000 pas.")


if __name__ == "__main__":
    main()
