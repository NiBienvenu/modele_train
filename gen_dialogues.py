#!/usr/bin/env python
"""
gen_dialogues.py : les conversations pour la phase "chat" du modèle zéro.

    python3 gen_dialogues.py --sortie A_METTRE_DANS_DRIVE/donnees_zero/dialogues

Sortie : des fichiers .jsonl au format {"messages": [...], "source": "..."}
et tests/dialogues_test.jsonl (10 % mis de côté, jamais vus à l'entraînement).

RÈGLE KIRUNDI : chaque phrase kirundi écrite ici DOIT exister telle quelle dans
data/cours_paires.jsonl (cours de Cox). Sinon le script s'arrête. On n'invente rien.
Le français, lui, est écrit librement.
"""

import argparse
import json
import random
import re
from pathlib import Path

ICI = Path(__file__).parent
COURS = [json.loads(l) for l in open(ICI / "data/cours_paires.jsonl", encoding="utf-8")]
RN_ATTESTE = {c["rn"].strip() for c in COURS}


def rn(phrase: str) -> str:
    """Renvoie la phrase kirundi seulement si elle est dans les cours de Cox."""
    if phrase not in RN_ATTESTE:
        raise ValueError(f"Phrase kirundi non trouvée dans les cours : {phrase!r}")
    return phrase


def conv(*tours, source):
    """conv("question", "réponse", "question 2", "réponse 2", source=...)"""
    roles = ["user", "assistant"] * len(tours)
    return {"messages": [{"role": r, "content": t} for r, t in zip(roles, tours)], "source": source}


def variantes(question: str):
    """Une question peut être écrite de plusieurs façons par les gens."""
    q = question.rstrip(" ?!.")
    return {question, q, q.lower(), q + " ?", q.lower() + " ?", q + "!"}


# ---------------------------------------------------------------------------
# 1. Salutations kirundi (Livre I, introduction)
# ---------------------------------------------------------------------------
def salutations_rn():
    src = "Cox Livre I, introduction"
    bonjour_matin = [rn("Bwakeye."), rn("Bwakeye neza.")]
    bonjour_soir = [rn("Mwiriwe."), rn("Mwiriwe neza.")]
    nouvelles = [rn("N'amaki?"), rn("N'amahoro?")]
    bien = [rn("N'amahoro."), rn("N'ameza.")]
    out = []
    for s in bonjour_matin + bonjour_soir:
        reponses = bonjour_matin if s in bonjour_matin else bonjour_soir
        for r in reponses:
            out.append(conv(s, r, source=src))
            for q in nouvelles:
                for b in bien:
                    out.append(conv(s, r, q, b, source=src))
                    out.append(conv(s, r, q, b, rn("Urakoze."), rn("N'agasaga."), source=src))
    for q in nouvelles:
        for b in bien:
            out.append(conv(q, b, source=src))
    out.append(conv(rn("Uragumye?"), rn("Ndagumye."), source=src))
    out.append(conv(rn("Ndagize bwakeye."), rn("Bwakeye neza."), source=src))
    out.append(conv(rn("N'agasaga."), rn("N'akagaruka."), source=src))
    return out


# ---------------------------------------------------------------------------
# 2. Questions en français SUR le kirundi (réponses tirées des cours)
# ---------------------------------------------------------------------------
def questions_sur_rn():
    src = "Cox Livre I"
    def m(p):  # la phrase sans le point final
        return rn(p).rstrip(".")

    agasaga, akagaruka, amaki = m("N'agasaga."), m("N'akagaruka."), rn("N'amaki?")
    amahoro = rn("N'amahoro.")
    faits = [
        ("Comment dit-on bonjour en kirundi ?",
         f"Le matin, on dit « {m('Bwakeye.')} » (ou « {m('Bwakeye neza.')} »). L'après-midi, le soir, "
         f"ou après la première rencontre du jour, on dit « {m('Mwiriwe.')} »."),
        ("Comment dit-on merci en kirundi ?",
         f"On dit « {m('Urakoze.')} » à une personne, et « Murakoze » à plusieurs personnes."),
        ("Comment dire au revoir en kirundi ?",
         f"On dit « {agasaga} ». Pour dire « à bientôt », on dit « {akagaruka} »."),
        ("Comment demander « comment ça va ? » en kirundi ?",
         f"On demande « {amaki} » (quelles nouvelles ?) ou « {rn('Uragumye?')} » (tu vas bien ?). "
         f"On répond « {amahoro} » ou « {rn('Ndagumye.')} »."),
        ("Comment dit-on « je ne sais pas » en kirundi ?", f"On dit « {rn('sinzi')} »."),
        ("Comment demander le nom de quelqu'un en kirundi ?", f"On demande « {rn('Witwa nde?')} »."),
        ("Comment saluer un supérieur le matin en kirundi ?",
         f"On dit « {m('Ndagize bwakeye.')} », c'est une forme de respect."),
    ]
    return [conv(v, r, source=src) for q, r in faits for v in variantes(q)]


# ---------------------------------------------------------------------------
# 3. Traductions (toutes les paires des cours de Cox)
# ---------------------------------------------------------------------------
def traductions_cours():
    out = []
    vers_rn = ["Traduis en kirundi : {}", "Comment dit-on en kirundi : {}", "En kirundi : {}"]
    vers_fr = ["Traduis en français : {}", "Que veut dire « {} » ?", "Que signifie « {} » en français ?"]
    for c in COURS:
        fr = re.sub(r"\s*\((?:mot à mot|lit\.)[^)]*\)", "", c["fr"]).strip()
        src = "Cox " + c["source"]
        reponse_fr = fr if not c.get("note") else f"{fr} ({c['note']})"
        out.append(conv(random.choice(vers_rn).format(fr), c["rn"], source=src))
        out.append(conv(random.choice(vers_fr).format(c["rn"]), reponse_fr, source=src))
    return out


# ---------------------------------------------------------------------------
# 4. Petite conversation en français (écrite librement)
# ---------------------------------------------------------------------------
PRESENTATION = ("Je suis un assistant qui parle français et kirundi. Je peux saluer, répondre à des "
                "questions simples et traduire des phrases simples entre le français et le kirundi.")
NE_SAIS_PAS = [
    "Je n'ai pas assez d'informations sur ce sujet pour répondre correctement.",
    "Je ne connais pas la réponse exacte. Je préfère ne pas inventer.",
    "Je n'ai pas cette information. Si vous avez un document qui en parle, je pourrai mieux vous aider.",
]
PETITES = [
    (["Bonjour", "Salut", "Bonjour !", "Coucou", "Bonsoir"], ["Bonjour ! Comment puis-je vous aider ?",
                                                             "Bonjour ! Que puis-je faire pour vous ?"]),
    (["Comment ça va ?", "Ça va ?", "Comment allez-vous ?", "Tu vas bien ?"],
     ["Je vais bien, merci. Et vous ?", "Très bien, merci ! Comment puis-je vous aider ?"]),
    (["Je vais bien", "Ça va bien, merci", "Bien, et toi ?"], ["Tant mieux ! Que puis-je faire pour vous ?"]),
    (["Merci", "Merci beaucoup", "Merci !"], ["Avec plaisir !", "De rien, c'est un plaisir."]),
    (["Au revoir", "À bientôt", "Bonne journée"], ["Au revoir, à bientôt !", "Bonne journée à vous !"]),
    (["Qui es-tu ?", "Tu es qui ?", "Présente-toi", "Que sais-tu faire ?", "Qu'est-ce que tu peux faire ?"], [PRESENTATION]),
    (["Tu parles kirundi ?", "Parles-tu kirundi ?", "Tu comprends le kirundi ?"],
     ["Oui, je parle kirundi et français. Je connais surtout les phrases simples et les salutations."]),
    (["Tu parles anglais ?", "Do you speak English?"],
     ["Je parle seulement français et kirundi."]),
    (["Quelle langue parle-t-on au Burundi ?", "Quelles sont les langues du Burundi ?"],
     ["Au Burundi, on parle surtout le kirundi. Le français est aussi utilisé, ainsi que l'anglais et le swahili."]),
    (["Quelle est la capitale du Burundi ?"],
     ["La capitale politique du Burundi est Gitega. Bujumbura est la capitale économique."]),
    (["Combien font 2 + 2 ?"], ["2 + 2 = 4."]),
    (["Combien font 10 fois 3 ?"], ["10 × 3 = 30."]),
    (["Quel jour sommes-nous ?", "Quelle heure est-il ?"],
     ["Je n'ai pas accès à l'heure ni à la date. Regardez votre téléphone ou votre ordinateur."]),
    (["Raconte-moi une blague"], ["Pourquoi les poissons n'aiment pas jouer au tennis ? Parce qu'ils ont peur du filet !"]),
    (["Tu peux m'aider ?", "J'ai une question", "Aide-moi"], ["Bien sûr. Posez votre question."]),
]
INCONNU = [
    "Que dit l'article 47 du décret numéro 100/215 ?",
    "Quel est le numéro de téléphone du ministère ?",
    "Qui a gagné le match d'hier ?",
    "Quel est le prix du sucre aujourd'hui à Bujumbura ?",
    "Quelle est l'adresse de mon voisin ?",
    "Que contient la page 230 du livre de mon école ?",
    "Combien d'habitants avait ma colline en 1950 ?",
    "Quel est le nom du directeur de cette entreprise ?",
    "Résume le document que j'ai écrit hier.",
    "Quelle note ai-je eue à l'examen ?",
]


def petite_conversation():
    out, src = [], "écrit (français)"
    for questions, reponses in PETITES:
        for q in questions:
            for v in variantes(q):
                out.append(conv(v, random.choice(reponses), source=src))
    for q in INCONNU:
        for v in variantes(q):
            out.append(conv(v, random.choice(NE_SAIS_PAS), source=src))
    # Plusieurs tours : le modèle doit suivre la conversation
    out.append(conv("Bonjour", "Bonjour ! Comment puis-je vous aider ?",
                    "Comment dit-on merci en kirundi ?", "On dit « Urakoze ».",
                    "Merci", "Avec plaisir !", source=src))
    out.append(conv("Salut", "Bonjour ! Que puis-je faire pour vous ?",
                    "Qui es-tu ?", PRESENTATION, "Au revoir", "Au revoir, à bientôt !", source=src))
    return out


# ---------------------------------------------------------------------------
# 5. Anciennes questions/réponses (déjà vérifiées), en petite quantité
# ---------------------------------------------------------------------------
def anciennes_qr(dossier_qr: Path, part: dict):
    out = []
    for nom, p in part.items():
        f = dossier_qr / f"{nom}.jsonl"
        if not f.exists():
            print("absent :", f)
            continue
        lignes = f.read_text(encoding="utf-8").splitlines()
        for l in random.sample(lignes, int(len(lignes) * p)):
            d = json.loads(l)
            out.append(conv(d["user"], d["assistant"], source=f"qr/{nom}"))
    return out


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--sortie", required=True)
    a.add_argument("--qr", default=str(ICI / "A_METTRE_DANS_DRIVE/donnees/qr"))
    args = a.parse_args()
    random.seed(42)
    sortie = Path(args.sortie)
    sortie.mkdir(parents=True, exist_ok=True)

    groupes = {
        "salutations_rn": salutations_rn(),
        "questions_sur_rn": questions_sur_rn(),
        "traductions_cours": traductions_cours(),
        "petite_conversation": petite_conversation(),
        # Part gardée de chaque ancien fichier : peu de tableaux, beaucoup de vraies phrases
        "anciennes_qr": anciennes_qr(Path(args.qr), {
            "cours": 1.0, "traduction_bible": 1.0, "lexique_rn": 0.3, "syntaxe_rn": 0.3,
            "accords_rn": 0.03, "conjugaison_rn": 0.03, "conjugaison_fr": 0.01}),
    }
    tests = []
    for nom, convs in groupes.items():
        random.shuffle(convs)
        n_test = max(1, len(convs) // 10)
        tests += [dict(c, categorie=nom) for c in convs[:n_test]]
        with open(sortie / f"{nom}.jsonl", "w", encoding="utf-8") as f:
            for c in convs[n_test:]:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(f"{nom:22s} {len(convs) - n_test:6d} entraînement | {n_test:5d} test")
    # Format de evaluer.py : on teste la DERNIÈRE réponse, les tours d'avant servent d'historique
    with open(ICI / "tests/dialogues_test.jsonl", "w", encoding="utf-8") as f:
        for c in tests:
            m = c["messages"]
            historique = "".join(f"<|user|> {x['content']} <|assistant|>" if x["role"] == "user"
                                 else f" {x['content']}<|endoftext|>" for x in m[:-2])
            test = {"categorie": c["categorie"], "question": m[-2]["content"], "attendu": m[-1]["content"],
                    "historique": historique, "vu": False, "source": c["source"]}
            f.write(json.dumps(test, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
