
# Advanced Resume Screening (Flask + Bootstrap + NER)

## Features
- Large dataset with job descriptions and resumes
- Precomputed embeddings for fast matching
- Stylish Bootstrap 5 UI (gradient navbar, cards, match score bars)
- Skill extraction using spaCy (NER)
- Matching skills highlighted in green
- Upload resume or paste job description → See top matches instantly

## Setup
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```
Run the app:
```bash
flask run
```
