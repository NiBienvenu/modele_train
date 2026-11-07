#!/usr/bin/env python
"""
train_tokenizer.py : crée NOTRE tokenizer français + kirundi.

MÉCANISME 1 : LE TOKENIZER.
Le tokenizer de Pythia a été fait pour l'anglais. Il coupe "baragiye" en
morceaux qui n'ont aucun sens. Ici, on apprend les VRAIS morceaux de nos
deux langues : préfixes kirundi (aba-, iki-, ba-ra-), terminaisons
françaises (-tion, -ement, -aient), mots fréquents entiers, etc.

Comment il apprend (algorithme BPE) :
  1) Au départ, chaque lettre (en fait chaque octet) est un morceau.
  2) On cherche la paire de morceaux qui revient le plus souvent dans
     tous les textes, par exemple "a" + "b" -> "ab".
  3) On la colle en un nouveau morceau, puis on recommence.
  4) On s'arrête quand on a "taille_vocab" morceaux.
Résultat : les suites fréquentes deviennent un seul morceau.

IMPORTANT : on crée le tokenizer UNE SEULE FOIS, sur TOUTES les données
(grammaire, livres, lois...). Ensuite, il ne doit plus jamais changer,
sinon le modèle ne reconnaît plus ses propres numéros.

Utilisation dans Colab :
    !python train_tokenizer.py \
        --source "/content/drive/MyDrive/donnees/**/*_fr*.txt=1" \
        --source "/content/drive/MyDrive/donnees/**/*_rn*.txt=3" \
        --sortie /content/drive/MyDrive/modele/tokenizer_fr_rn

Le "=3" veut dire : lire ces fichiers 3 fois. Le kirundi a moins de
textes que le français, on le répète pour qu'il ait assez de place dans
le vocabulaire.
"""

import argparse
import glob
import os
import unicodedata

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

# Les mots spéciaux. Ils doivent être les mêmes que dans train.py.
SPECIAUX = ["<|endoftext|>", "<pad>", "<|user|>", "<|assistant|>"]

# Phrases pour vérifier le résultat à la fin.
PHRASES_TEST = [
    ("fr", "Les filles que j'ai vues sont parties à l'école hier."),
    ("fr", "Nous mangerions si vous aviez préparé le repas."),
    ("fr", "Le présent décret entre en vigueur le jour de sa signature."),
    ("rn", "Abana baragiye ku ishure."),
    ("rn", "Ejo nzokora akazi kanje."),
    ("rn", "Umwana mwiza yubaha abavyeyi biwe."),
]


def lire_sources(sources):
    """
    Lit les fichiers et donne le texte ligne par ligne au tokenizer.
    Chaque source est répétée selon son poids ("chemin=poids").
    """
    for source in sources:
        chemin, _, poids = source.rpartition("=")
        if not chemin:  # pas de "=" : poids 1
            chemin, poids = source, "1"
        fichiers = sorted(glob.glob(chemin, recursive=True))
        if not fichiers:
            print(f"ATTENTION : aucun fichier trouvé pour {chemin}")
            continue
        print(f"{len(fichiers)} fichier(s) pour {chemin} (lus {poids} fois)")
        for _ in range(int(poids)):
            for f in fichiers:
                with open(f, encoding="utf-8", errors="ignore") as fh:
                    for ligne in fh:
                        ligne = ligne.strip()
                        if ligne:
                            yield ligne


def construire_tokenizer():
    """Prépare un tokenizer vide, prêt à apprendre."""
    tok = Tokenizer(models.BPE())

    # NFC : "é" peut s'écrire de 2 façons dans l'ordinateur (une lettre,
    # ou "e" + accent). NFC met tout sous la même forme. Sans ça, le
    # modèle croirait que ce sont deux lettres différentes.
    tok.normalizer = normalizers.NFC()

    # Byte-level : n'importe quel caractère peut être écrit (même les
    # symboles rares). Il n'y a donc jamais de mot "inconnu".
    # add_prefix_space=False : on garde le texte tel quel.
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    return tok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", action="append", required=True,
                   help='"chemin/avec/*.txt=poids" (on peut en mettre plusieurs)')
    p.add_argument("--taille_vocab", type=int, default=24000)
    p.add_argument("--freq_min", type=int, default=2,
                   help="un morceau doit apparaître au moins ce nombre de fois")
    p.add_argument("--sortie", required=True)
    args = p.parse_args()

    tok = construire_tokenizer()
    entraineur = trainers.BpeTrainer(
        vocab_size=args.taille_vocab,
        min_frequency=args.freq_min,
        special_tokens=SPECIAUX,      # numéros 0, 1, 2, 3
        # On part des 256 octets de base : toutes les lettres existent déjà.
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    print("Apprentissage des morceaux (BPE)...")
    tok.train_from_iterator(lire_sources(args.source), trainer=entraineur)

    # On "emballe" le tokenizer au format Hugging Face, pour que
    # AutoTokenizer.from_pretrained(...) dans train.py puisse le lire.
    tok_hf = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        unk_token="<|endoftext|>",
        pad_token="<pad>",
        additional_special_tokens=["<|user|>", "<|assistant|>"],
        model_max_length=2048,
    )
    os.makedirs(args.sortie, exist_ok=True)
    tok_hf.save_pretrained(args.sortie)
    print(f"\nTokenizer sauvegardé dans {args.sortie} ({len(tok_hf)} morceaux)\n")

    # ---- Vérification --------------------------------------------------
    # "Morceaux par mot" : plus c'est proche de 1, mieux c'est.
    # Pythia (anglais) donne souvent 3 à 5 pour le kirundi.
    # Un bon tokenizer fr+rn devrait donner autour de 1.3 à 2.
    print("Vérification :")
    for langue, phrase in PHRASES_TEST:
        phrase = unicodedata.normalize("NFC", phrase)
        morceaux = tok_hf.tokenize(phrase)
        ratio = len(morceaux) / len(phrase.split())
        lisibles = [tok_hf.convert_tokens_to_string([m]) for m in morceaux]
        print(f"[{langue}] {ratio:.2f} morceaux/mot : {lisibles}")

    # Les mots spéciaux doivent rester entiers (un seul numéro chacun).
    test = tok_hf("<|user|> Bonjour <|assistant|> Amahoro<|endoftext|>")["input_ids"]
    assert tok_hf.convert_tokens_to_ids("<|user|>") in test, "<|user|> est coupé !"
    assert tok_hf.eos_token_id in test, "<|endoftext|> est coupé !"
    print("\nMots spéciaux : OK")


if __name__ == "__main__":
    main()
