"""Construit datasets/dataset_deux/ (modèle DEUX) à partir de docs/docs 3.

    python3 collecte/lire_docs3.py "../docs/docs 3"   (PDF numériques -> data/docs3/)
    python3 preparer_donnees_deux.py
Entrées :
  data/docs3/                      textes des PDF numériques + articles des constitutions
  data/scans_transcrits/Indimburo  grammaire (recopiée à la main, page par page)
  data/scans_transcrits/Dusome5    livre de lecture 5e année
  data/scans_transcrits/N21        loi n°1/21 du 12/07/2024 sur la presse (FR | RN)
  data/indimburo_qa/*.jsonl        questions-réponses de grammaire tirées du livre
Une ligne avec [?] (mot douteux) ne va JAMAIS dans l'entraînement : elle va dans a_verifier.tsv.
"""

import json
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

ICI = Path(__file__).resolve().parent
DATA = ICI / "data"
SCANS = DATA / "scans_transcrits"
OUT = ICI / "datasets/dataset_deux"
random.seed(2)

FR_MOTS = set("le la les des du de et est un une en au aux pour par sur dans que qui ne pas son sa ses il elle "
              "sont être ou ce cette à l d qu toute tout tous".split())
RN_MOTS = set("na mu ku canke kandi ni ari uwo iyo ivyo bwo rwo vyo ryo kugira kuri n y w kw mw ata "
              "umuntu abantu ico ibi iki".split())

a_verifier = []


def langue(t: str) -> str:
    mots = re.findall(r"[a-zà-ÿ]+", t.lower())
    fr = sum(m in FR_MOTS for m in mots)
    rn = sum(m in RN_MOTS for m in mots)
    return "fr" if fr > rn else "rn"


def pages(livre: str) -> list:
    """[(page, [lignes sûres])] ; les lignes avec [?] vont dans a_verifier."""
    out = []
    for f in sorted((SCANS / livre).glob("p-*.txt")):
        sures = []
        for l in f.read_text(encoding="utf-8").splitlines():
            l = l.strip()
            if not l or (l.startswith("[") and l.endswith("]")):
                continue
            if "[?]" in l or "[...]" in l:
                a_verifier.append((livre, f.stem, l))
                continue
            sures.append(l.replace("’", "'"))
        out.append((f.stem, sures))
    return out


def ecrire_txt(chemin: Path, lignes: list):
    chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_text("\n".join(lignes).strip() + "\n", encoding="utf-8")


def ecrire_jsonl(chemin: Path, objets: list):
    chemin.parent.mkdir(parents=True, exist_ok=True)
    with chemin.open("w", encoding="utf-8") as f:
        for o in objets:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")


def paragraphes(lignes: list, source: str, taille=1200) -> list:
    """Regroupe les lignes en morceaux de ~taille caractères pour lecture.jsonl."""
    docs, cur = [], ""
    for l in lignes:
        if cur and len(cur) + len(l) > taille:
            docs.append({"text": cur, "source": source}); cur = ""
        cur = f"{cur}\n{l}" if cur else l
    if cur:
        docs.append({"text": cur, "source": source})
    return docs


def phrases(t: str) -> list:
    """Coupe en phrases (après . ; ! ?), sans couper « 1° » ni les sigles courts."""
    t = re.sub(r"\s+", " ", t).strip()
    return [x.strip() for x in re.split(r"(?<=[.!?;])\s+(?=[A-ZÀ-ÝÉ])", t) if len(x.split()) >= 3]


# ---------- Bible : choisir les meilleurs versets (correction 2) ----------
GENEALOGIE = re.compile(r"\b(engendra|fils de|fille de|yavyaye|mwene|bene|descendants|clans?|recens)", re.I)


def note_verset(fr: str, rn: str) -> float:
    """Plus haut = meilleur pour apprendre : passage narratif, peu de noms propres, pas de liste.
    Run deux : la limite à 45 mots et l'idéal à 18 mots gardaient des versets trop courts
    (médiane 20 mots) alors que les tests Bible font 38 mots en médiane -> chrF 29 -> 16."""
    mots = fr.split()
    n = len(mots)
    if n < 6 or n > 120:
        return -1
    noms = sum(1 for i, m in enumerate(mots) if i and m[:1].isupper())
    note = 1.0 - 2.5 * noms / n                         # beaucoup de noms propres = liste ou généalogie
    note -= 0.6 * len(GENEALOGIE.findall(fr + " " + rn))
    note -= 0.4 * len(re.findall(r"\d", fr))
    note -= 0.01 * abs(n - 35)                          # idéal vers 35 mots (comme les tests)
    if not 0.5 < len(rn) / max(len(fr), 1) < 2.0:
        note -= 1
    return note + random.random() * 0.15                # un peu de hasard pour varier les livres


def bible_choisie(n=16000) -> list:
    """Run deux : 9000 versets, 1/3 seulement vers le français, alors que 58 % des tests
    sont kirundi -> français. Maintenant : 16000 passages, moitié dans chaque sens."""
    vers_fr, vers_rn = [], []
    tests = {json.loads(l)["question"] for l in (ICI / "tests/dialogues_test.jsonl").open(encoding="utf-8")}
    for l in (ICI / "datasets/dataset/dialogues/traductions_bible.jsonl").open(encoding="utf-8"):
        o = json.loads(l)
        q, r = o["messages"][0]["content"], o["messages"][1]["content"]
        if q in tests:                                  # jamais une question de test dans l'entraînement
            continue
        en_fr = "français" in q.split(" : ", 1)[0]
        fr, rn = (r, q) if en_fr else (q, r)
        fr, rn = fr.split(" : ", 1)[-1], rn.split(" : ", 1)[-1]
        (vers_fr if en_fr else vers_rn).append((note_verset(fr, rn), o))
    choix = []
    for groupe in (vers_fr, vers_rn):
        groupe.sort(key=lambda x: -x[0])
        choix += [o for note, o in groupe[:n // 2] if note > 0]
    random.shuffle(choix)
    return choix


# ---------- constitutions : apprendre ET mémoriser ----------
Q_RN = ["Ingingo ya {n} y'{nom} ivuga iki?", "Subiramwo ingingo ya {n} y'{nom}.",
        "Mbwira ingingo ya {n} y'{nom} uko yanditse."]
Q_FR = ["Que dit l'article {n} de {nomfr} ? Réponds en kirundi.", "Récite l'article {n} de {nomfr} en kirundi."]


def constitution(an: str, nom: str, nomfr: str, arts: dict) -> tuple:
    txt, memo = [f"Ibwirizwa shingiro rya Repubulika y'Uburundi ({an})", ""], []
    for n in sorted(arts, key=int):
        v = arts[n]
        rn = v["rn"] if isinstance(v, dict) else v
        fr = v.get("fr", "") if isinstance(v, dict) else ""
        txt += [f"Ingingo ya {n}", rn, ""]
        src = f"constitution_{an} art.{n}"
        for q in Q_RN:
            memo.append({"user": q.format(n=n, nom=nom), "assistant": rn, "source": src})
        for q in Q_FR:
            memo.append({"user": q.format(n=n, nomfr=nomfr), "assistant": rn, "source": src})
        memo.append({"user": f"Iyi nteruro iva mu ngingo iyihe y'{nom}? « {rn[:160]} »",
                     "assistant": f"Ni ingingo ya {n} y'{nom}.", "source": src})
        if fr:
            memo.append({"user": f"Que dit l'article {n} de {nomfr} ?", "assistant": fr, "source": src})
    return txt, memo


def main():
    lecture = []

    # 1. textes numériques de docs 3
    for f in sorted((DATA / "docs3").glob("*.txt")):
        if f.stem.startswith("constitution"):
            continue
        lignes = [l for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        ecrire_txt(OUT / f"lecture_rn/{f.stem}.txt", lignes)
        lecture += paragraphes(lignes, f.stem)

    # 2. constitutions (séparées)
    a05 = json.loads((DATA / "docs3/constitution_2005.json").read_text(encoding="utf-8"))
    a18 = json.loads((DATA / "docs3/constitution_2018.json").read_text(encoding="utf-8"))
    t05, m05 = constitution("2005", "Ibwirizwa shingiro rya 2005", "la Constitution du Burundi de 2005", a05)
    t18, m18 = constitution("2018", "Ibwirizwa shingiro rya 2018", "la Constitution du Burundi de 2018", a18)
    ecrire_txt(OUT / "constitutions/constitution_2005.txt", t05)
    ecrire_txt(OUT / "constitutions/constitution_2018.txt", t18)
    ecrire_txt(OUT / "constitutions/constitution_2018_fr.txt",
               sum(([f"Article {n}", a18[n]["fr"], ""] for n in sorted(a18, key=int) if a18[n]["fr"]), []))
    ecrire_jsonl(OUT / "constitutions/memoire_2005.jsonl", m05)
    ecrire_jsonl(OUT / "constitutions/memoire_2018.jsonl", m18)
    lecture += paragraphes(t05, "constitution_2005") + paragraphes(t18, "constitution_2018")

    # 3. traduction FR <-> RN : constitution 2018 + loi presse N21
    paires = []
    n_phrases = 0
    for n, v in a18.items():
        if not v["fr"]:
            continue
        # Article entier, puis phrase par phrase quand FR et RN ont le même nombre de phrases
        paires.append((v["fr"], v["rn"], f"constitution_2018 art.{n}"))
        fr_p, rn_p = phrases(v["fr"]), phrases(v["rn"])
        if len(fr_p) > 1 and len(fr_p) == len(rn_p):
            for i, (a, b) in enumerate(zip(fr_p, rn_p)):
                if 0.4 < len(b) / max(len(a), 1) < 2.5:     # longueurs trop différentes = mal aligné
                    paires.append((a, b, f"constitution_2018 art.{n} phrase {i + 1}"))
                    n_phrases += 1
    print(f"constitution 2018 : {n_phrases} paires phrase par phrase en plus des articles")
    n21_rn = []
    for page, lignes in pages("N21"):
        fr = [l for l in lignes if langue(l) == "fr"]
        rn = [l for l in lignes if langue(l) == "rn"]
        n21_rn += rn
        if len(fr) == len(rn):
            paires += [(a, b, f"loi_presse_2024 {page}") for a, b in zip(fr, rn)]
    trad = []
    for fr, rn, src in paires:
        trad.append({"user": f"Traduis en kirundi : {fr}", "assistant": rn, "source": src})
        trad.append({"user": f"Hindura mu gifaransa : {rn}", "assistant": fr, "source": src})
    ecrire_jsonl(OUT / "traduction/fr_rn.jsonl", trad)
    bible = bible_choisie()
    ecrire_jsonl(OUT / "traduction/bible_choisie.jsonl", bible)
    print(f"Bible : {len(bible)} passages choisis (moitié vers le français, longueur comme les tests)")
    ecrire_txt(OUT / "lecture_rn/loi_presse_2024.txt", n21_rn)
    lecture += paragraphes(n21_rn, "loi_presse_2024")

    # 4. Dusome 5 + ses insiguro
    d5, insiguro, dans = [], [], False
    for page, lignes in pages("Dusome5"):
        for l in lignes:
            if l.lower().startswith("insiguro y'amajambo"):
                dans = True
                continue
            m = re.match(r"^([^:]{2,40}?)\s*:\s*(.{3,})$", l)
            if dans and m:
                mot, sens = m.group(1).strip(), m.group(2).strip()
                for q in ["Que veut dire « {m} » en kirundi ?", "« {m} » bisobanura iki?"]:
                    insiguro.append({"user": q.format(m=mot), "assistant": f"« {mot} » : {sens}",
                                     "source": f"Dusome5 {page}"})
                continue
            dans = False
            if not re.fullmatch(r"\d+", l):
                d5.append(l)
    ecrire_txt(OUT / "lecture_rn/dusome5.txt", d5)
    ecrire_jsonl(OUT / "insiguro/dusome5_insiguro.jsonl", insiguro)
    lecture += paragraphes(d5, "dusome5")

    # 5. Indimburo : la grammaire (texte complet + questions-réponses)
    ind = []
    for page, lignes in pages("Indimburo"):
        ind += [f"[{page}]"] + lignes + [""]
    ecrire_txt(OUT / "ikibonezamvugo/indimburo_yikirundi.txt", ind)
    lecture += paragraphes([l for l in ind if not l.startswith("[p-")], "indimburo")
    qa, vus = [], set()
    for f in sorted((DATA / "indimburo_qa").glob("*.jsonl")):
        for l in f.read_text(encoding="utf-8").splitlines():
            try:
                o = json.loads(l)
            except json.JSONDecodeError:
                continue
            if "[?]" in o.get("user", "") + o.get("assistant", "") or o.get("user") in vus:
                continue
            vus.add(o["user"])
            qa.append(o)
    ecrire_jsonl(OUT / "ikibonezamvugo/ikibonezamvugo_qa.jsonl", qa)

    # 6. faits simples sur le Burundi (lecture + questions-réponses)
    faits = [l for l in (DATA / "faits_burundi.txt").read_text(encoding="utf-8").split("\n\n") if l.strip()]
    ecrire_txt(OUT / "faits/faits_burundi.txt", [f + "\n" for f in faits])
    shutil.copy2(DATA / "faits_burundi.jsonl", OUT / "faits/faits_burundi.jsonl")
    lecture += [{"text": f, "source": "faits_burundi"} for f in faits]

    # 7. FLORES : partie "dev" pour APPRENDRE ; "devtest" reste réservé aux tests (le même que modèle un)
    flores = [json.loads(l) for l in (DATA / "flores_fr_rn.jsonl").open(encoding="utf-8")]
    dev = [p for p in flores if p["partie"] == "dev"]
    ecrire_jsonl(OUT / "traduction/flores_dev.jsonl",
                 [o for p in dev for o in ({"user": f"Traduis en kirundi : {p['fr']}", "assistant": p["rn"]},
                                           {"user": f"Traduis en français : {p['rn']}", "assistant": p["fr"]})])
    tests_un = ICI / "datasets/dataset_un/tests/flores.jsonl"
    (OUT / "tests").mkdir(exist_ok=True)
    shutil.copy2(tests_un, OUT / "tests/flores.jsonl")          # EXACTEMENT les mêmes tests que le modèle un
    ids_test = {json.loads(l)["source"].split()[-1] for l in tests_un.open(encoding="utf-8")}
    assert not ids_test & {p["id"] for p in dev}, "un test FLORES est dans l'entraînement !"

    # 8. tout ce qui se lit, un seul fichier
    ecrire_jsonl(OUT / "lecture.jsonl", lecture)
    ecrire_txt(OUT / "lecture.txt", [d["text"] + "\n" for d in lecture])
    a_verifier.sort()
    (OUT / "a_verifier.tsv").write_text("livre\tpage\tligne\n" + "".join(f"{a}\t{b}\t{c}\n" for a, b, c in a_verifier),
                                        encoding="utf-8")

    mots = sum(len(d["text"].split()) for d in lecture)
    print(f"lecture: {len(lecture)} morceaux, {mots} mots")
    print(f"constitutions: 2005={len(a05)} ingingo, 2018={len(a18)} ingingo, mémoire={len(m05) + len(m18)} questions")
    print(f"traduction: {len(paires)} paires -> {len(trad)} exemples")
    print(f"FLORES dev: {len(dev)} paires (apprentissage) ; faits Burundi: {len(faits)}")
    print(f"insiguro Dusome5: {len(insiguro)} ; grammaire QA: {len(qa)} ; à vérifier: {len(a_verifier)} lignes")


def code_et_notebooks():
    code = OUT / "code"
    for f in ["train.py", "evaluer.py", "comparer_resultats.py", "requirements.txt", "configs/deux_lecture.yaml",
              "configs/deux_chat.yaml", "collecte/telecharger_publics.py", "tests/dialogues_test.jsonl"]:
        (code / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ICI / f, code / f)
    subprocess.run([sys.executable, str(ICI / "faire_notebooks_deux.py")], check=True)


if __name__ == "__main__":
    main()
    code_et_notebooks()
