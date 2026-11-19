r"""Extrait le texte des PDF numériques de docs/docs 3 vers data/docs3/.

    python3 collecte/lire_docs3.py ../docs/docs\ 3
Sorties (data/docs3/) :
  <nom>.txt                    texte kirundi propre, un paragraphe par ligne
  constitution_2005.json       {numéro: texte kirundi}
  constitution_2018.json       {numéro: {"rn": ..., "fr": ...}}
Les constitutions sont sur 2 colonnes : on lit chaque moitié de page séparément (pdftotext -x -W).
Les PDF scannés (Dusome5, Indimburo, N°21) sont recopiés à la main : data/scans_transcrits/.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

ICI = Path(__file__).resolve().parent.parent
SORTIE = ICI / "data/docs3"

SIMPLES = {
    "amategeko_inyifato_akazi.pdf": "amategeko_inyifato_akazi",
    "synthese_du_rapport-republika_yuburundi_ishusho_yiterambere_ridakumira-kirundi_version.pdf": "ishusho_iterambere_ridakumira",
    "Vision_BPE2040&PD2060_Kirundi.pdf": "icerekezo_2040_2060",
    "IJAMBO-RISHIKIRIJWE-NA-NYENICUBAHIRO-UMUKURU-W-IGIHUGU-I-MUGERA.pdf": "ijambo_mugera",
    "INTSINZI-2019.pdf": "intsinzi_2019",
    "CP Burundi _ Kirundi _ Jean Bigirimana_0.pdf": "itangazo_rsf",
}
C2005 = "BDI_Constitution_2005_RN.pdf"
C2018 = "Constitution 070618.pdf"


def pages(pdf: Path) -> tuple:
    info = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
    n = int(re.search(r"Pages:\s+(\d+)", info).group(1))
    w, h = map(float, re.search(r"Page size:\s+([\d.]+) x ([\d.]+)", info).groups())
    return n, w, h


def texte(pdf: Path, page=None, x=None, largeur=None, hauteur=None) -> str:
    cmd = ["pdftotext", "-q", "-layout", "-enc", "UTF-8"]
    if page:
        cmd += ["-f", str(page), "-l", str(page)]
    if x is not None:
        cmd += ["-x", str(int(x)), "-y", "0", "-W", str(int(largeur)), "-H", str(int(hauteur))]
    return subprocess.run(cmd + [str(pdf), "-"], capture_output=True, text=True).stdout


def nettoyer(brut: str) -> list:
    """Retire numéros de page, lignes de points, et recolle les lignes en paragraphes."""
    paras, cur = [], []
    for l in brut.replace("\f", "\n").split("\n"):
        l = re.sub(r"\s+", " ", l.replace("­", "")).strip()
        bruit = (not l or re.fullmatch(r"[\d\s.•…ivxlIVXL-]{0,12}", l) or re.search(r"\.{5,}", l)
                 or re.search(r"_{5,}", l) or re.search(r",\s*\d+(,\s*\d+)*\s*$", l)
                 or l.startswith("Igitabu c'amategeko") or l.startswith("Igitabu c’amategeko")
                 or re.fullmatch(r"18 Ntwarante 2005 . Ibwirizwa", l))
        if bruit:
            if cur:
                paras.append(" ".join(cur)); cur = []
            continue
        tete = re.match(r"^(l|I)ngingo ya \d+\s*:?$|^Article (\d+|premier)\s*:?$|^\d+(\.\d+)*\.\s", l)
        if tete and cur:
            paras.append(" ".join(cur)); cur = []
        if cur and cur[-1].endswith("-") and l[:1].islower():
            cur[-1] = cur[-1][:-1] + l
        else:
            cur.append(l)
        if re.match(r"^(l|I)ngingo ya \d+\s*:?$|^Article (\d+|premier)\s*:?$", l):
            paras.append(" ".join(cur)); cur = []
    if cur:
        paras.append(" ".join(cur))
    out = []
    for p in paras:
        p = p.replace("’", "'").replace("‘", "'")
        p = re.sub(r"\blngingo\b", "Ingingo", p)
        if len(p.split()) >= 2 or re.match(r"(Ingingo|Article|Umutwe|UMUTWE|IGICE|Igice)", p):
            out.append(p)
    return out


TETE = re.compile(r"^(Ingingo ya|Article)\s+(\d+|premier)\s*:?$", re.I)


def articles(paras: list) -> dict:
    """{numéro: texte} en coupant aux lignes « Ingingo ya N » / « Article N »."""
    arts, num, cur = {}, None, []
    for p in paras:
        m = TETE.match(p)
        if m:
            if num and cur:
                arts.setdefault(num, " ".join(cur))
            num = "1" if m.group(2).lower() == "premier" else m.group(2)
            cur = []
        elif num:
            if re.match(r"^(UMUTWE|IGICE|TITRE|CHAPITRE|Umutwe|Igice|Titre|Chapitre|Section|Ikigabane)\b", p):
                if cur:
                    arts.setdefault(num, " ".join(cur)); cur = []
                num = None
                continue
            cur.append(p)
    if num and cur:
        arts.setdefault(num, " ".join(cur))
    return arts


def colonnes(pdf: Path) -> tuple:
    n, w, h = pages(pdf)
    gauche, droite = [], []
    for p in range(1, n + 1):
        gauche.append(texte(pdf, p, 0, w / 2, h))
        droite.append(texte(pdf, p, w / 2, w / 2, h))
    return gauche, droite


def main(dossier: Path):
    SORTIE.mkdir(parents=True, exist_ok=True)
    for pdf, nom in SIMPLES.items():
        paras = nettoyer(texte(dossier / pdf))
        (SORTIE / f"{nom}.txt").write_text("\n".join(paras) + "\n", encoding="utf-8")
        print(f"{nom}: {len(paras)} paragraphes, {sum(len(p.split()) for p in paras)} mots")

    # 2005 : kirundi seul, 2 colonnes -> gauche puis droite, page par page
    g, d = colonnes(dossier / C2005)
    paras = nettoyer("\n\n".join(a + "\n\n" + b for a, b in zip(g, d)))
    (SORTIE / "constitution_2005.txt").write_text("\n".join(paras) + "\n", encoding="utf-8")
    a05 = articles(paras)
    (SORTIE / "constitution_2005.json").write_text(json.dumps(a05, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"constitution_2005: {len(a05)} ingingo")

    # 2018 : français à gauche, kirundi à droite
    g, d = colonnes(dossier / C2018)
    fr, rn = nettoyer("\n\n".join(g)), nettoyer("\n\n".join(d))
    (SORTIE / "constitution_2018.txt").write_text("\n".join(rn) + "\n", encoding="utf-8")
    (SORTIE / "constitution_2018_fr.txt").write_text("\n".join(fr) + "\n", encoding="utf-8")
    afr, arn = articles(fr), articles(rn)
    a18 = {k: {"rn": arn[k], "fr": afr.get(k, "")} for k in sorted(arn, key=int)}
    (SORTIE / "constitution_2018.json").write_text(json.dumps(a18, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"constitution_2018: {len(arn)} ingingo, {sum(1 for v in a18.values() if v['fr'])} avec le français")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else ICI.parent / "docs/docs 3"))
