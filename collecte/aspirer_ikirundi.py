"""Collecte le texte de toutes les pages de https://www.ikirundi.org/

    python3 collecte/aspirer_ikirundi.py data/ikirundi_org.jsonl

- Visite le site page par page (1 page par seconde, pour ne pas le surcharger).
- Garde seulement le texte de l'article (pas les menus).
- Reprend là où il s'est arrêté si on le relance.
Sortie : une ligne JSON par page : {"url", "titre", "texte"}.
"""

import html
import json
import re
import sys
import time
import urllib.request
from collections import deque
from html.parser import HTMLParser
from pathlib import Path

SITE = "https://www.ikirundi.org"
PAUSE = 1.0
IGNORER = re.compile(r"\.(jpe?g|png|gif|pdf|mp3|mp4|zip|css|js)$|^/(media|templates|images|ejo_images)/|index\.php", re.I)


class Article(HTMLParser):
    """Lit le texte du bloc <div class="com-content-article__body"> et les liens de la page."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.liens, self.morceaux, self.titre = set(), [], ""
        self._prof = 0  # profondeur des <div> dans l'article (0 = dehors)
        self._ignorer = 0  # dans <script>/<style>
        self._dans_titre = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "a" and a.get("href", "").startswith("/"):
            self.liens.add(a["href"].split("#")[0].split("?")[0])
        if tag in ("script", "style"):
            self._ignorer += 1
        if tag == "title":
            self._dans_titre = True
        if tag == "div":
            if self._prof:
                self._prof += 1
            elif "com-content-article__body" in (a.get("class") or ""):
                self._prof = 1
        if self._prof and tag in ("br", "p", "li", "tr", "h1", "h2", "h3", "h4", "div"):
            self.morceaux.append("\n")
        elif self._prof and tag in ("td", "th", "span", "a", "img"):
            self.morceaux.append(" ")  # sinon deux cases de tableau collent leurs mots

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._ignorer -= 1
        if tag == "title":
            self._dans_titre = False
        if tag == "div" and self._prof:
            self._prof -= 1
        if self._prof and tag in ("td", "th", "span", "a"):
            self.morceaux.append(" ")

    def handle_data(self, data):
        if self._dans_titre:
            self.titre += data
        elif self._prof and not self._ignorer:
            self.morceaux.append(data)

    def texte(self) -> str:
        t = html.unescape("".join(self.morceaux))
        t = t.split("-->")[0]  # après "-->" commence le menu de la rubrique
        lignes = [re.sub(r"[ \t\xa0]+", " ", l).strip() for l in t.splitlines()]
        return "\n".join(l for l in lignes if l)


def telecharger(url: str, essais: int = 3) -> str:
    """Le site est parfois lent : on réessaie 3 fois, en attendant de plus en plus."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (corpus kirundi, usage personnel)"})
    for i in range(essais):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read().decode("utf-8", "replace")
        except Exception:
            if i == essais - 1:
                raise
            time.sleep(10 * (i + 1))


def main(sortie: Path):
    deja = set()
    if sortie.exists():
        deja = {json.loads(l)["url"] for l in sortie.open(encoding="utf-8")}
    vus, file = set(deja), deque(["/"])
    vus.add("/")
    # Les pages déjà sauvées sont relues pour retrouver leurs liens
    file.extend(sorted(u.removeprefix(SITE) for u in deja))
    with sortie.open("a", encoding="utf-8") as f:
        while file:
            chemin = file.popleft()
            url = SITE + chemin
            try:
                page = telecharger(url)
            except Exception as e:
                print("ERREUR", url, e, file=sys.stderr)
                continue
            time.sleep(PAUSE)
            p = Article()
            p.feed(page)
            for lien in p.liens:
                if lien not in vus and not IGNORER.search(lien):
                    vus.add(lien)
                    file.append(lien)
            texte = p.texte()
            if url not in deja and len(texte) > 30:
                f.write(json.dumps({"url": url, "titre": p.titre.strip(), "texte": texte}, ensure_ascii=False) + "\n")
                f.flush()
                deja.add(url)
                print(len(deja), url, len(texte), flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
