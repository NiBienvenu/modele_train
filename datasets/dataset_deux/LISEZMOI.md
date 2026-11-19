# dataset_deux : ce que le modèle DEUX apprend en plus (docs/docs 3)

Refaire ce dossier, dans modele_train/ :
    python3 collecte/lire_docs3.py "../docs/docs 3"
    python3 preparer_donnees_deux.py

- `lecture.txt` / `lecture.jsonl` : tout le kirundi à lire (morceaux de ~1200 caractères, `{"text","source"}`)
- `lecture_rn/` : un .txt par document (discours, Vision 2040-2060, rapport, règlement, communiqué RSF, Dusome5, loi presse 2024)
- `constitutions/` : 2005 et 2018 GARDÉES À PART
  - `constitution_2005.txt`, `constitution_2018.txt` (kirundi), `constitution_2018_fr.txt`
  - `memoire_2005.jsonl`, `memoire_2018.jsonl` : questions-réponses pour apprendre chaque ingingo par cœur
    (« Ingingo ya 26 ... ivuga iki? » -> texte exact). Pour la mémoire, entraîner plusieurs époques sur ces fichiers.
  - Le PDF de 2005 est un extrait : 151 ingingo seulement. 2018 : 289 sur 292 (110, 132, 187 manquent).
- `ikibonezamvugo/` : la GRAMMAIRE (Indimburo y'ikirundi, 76 pages recopiées à la main)
  - `indimburo_yikirundi.txt` : le livre entier, page par page ([p-NN])
  - `ikibonezamvugo_qa.jsonl` : questions-réponses tirées du livre seulement (définitions, règles, tableaux, exercices)
- `insiguro/dusome5_insiguro.jsonl` : sens des mots difficiles de Dusome 5
- `traduction/fr_rn.jsonl` : paires FR <-> RN (constitution 2018 article par article + loi presse 2024)
- `a_verifier.tsv` : lignes avec un mot douteux [?]. PAS dans l'entraînement. Corriger dans
  data/scans_transcrits/<livre>/p-NN.txt (enlever [?]) puis relancer le script.

Les scans (Indimburo, Dusome5, N°21) ont été lus image par image, pas par OCR : Tesseract ne connaît pas le kirundi.
