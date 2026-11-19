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
    paires = [(v["fr"], v["rn"], f"constitution_2018 art.{n}") for n, v in a18.items() if v["fr"]]
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

    # 6. tout ce qui se lit, un seul fichier
    ecrire_jsonl(OUT / "lecture.jsonl", lecture)
    ecrire_txt(OUT / "lecture.txt", [d["text"] + "\n" for d in lecture])
    a_verifier.sort()
    (OUT / "a_verifier.tsv").write_text("livre\tpage\tligne\n" + "".join(f"{a}\t{b}\t{c}\n" for a, b, c in a_verifier),
                                        encoding="utf-8")

    mots = sum(len(d["text"].split()) for d in lecture)
    print(f"lecture: {len(lecture)} morceaux, {mots} mots")
    print(f"constitutions: 2005={len(a05)} ingingo, 2018={len(a18)} ingingo, mémoire={len(m05) + len(m18)} questions")
    print(f"traduction: {len(paires)} paires -> {len(trad)} exemples")
    print(f"insiguro Dusome5: {len(insiguro)} ; grammaire QA: {len(qa)} ; à vérifier: {len(a_verifier)} lignes")


if __name__ == "__main__":
    main()
