# Cosmian AI clients

How to interact with a AI runner backend

## Summarize

Call endpoint `/summarize` with `src_lang="en"`

```bash
python summarize.py http://localhost:5000 ../sample_data/sample_en_doc.txt
```

## Translate

Call endpoint `/translate` with `src_lang="en"` and `tgt_lang="fr"`

```bash
python translate.py http://localhost:5000 ../sample_data/sample_en_doc.txt
```
