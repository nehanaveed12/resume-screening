
import pandas as pd
import numpy as np
import os
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class ResumeMatcher:
    def __init__(self, data_path="sample_data", model_name="all-MiniLM-L6-v2", use_precomputed=False):
        self.data_path = data_path
        self.model = SentenceTransformer(model_name)
        self.resumes = pd.read_csv(os.path.join(data_path, "resumes.csv"))
        self.jobs = pd.read_csv(os.path.join(data_path, "jobs.csv"))
        self.use_precomputed = use_precomputed
        self.nlp = spacy.load("en_core_web_sm")

        if self.use_precomputed:
            self.resume_embeddings = np.load(os.path.join(data_path, "resume_embeddings.npy"))
            self.job_embeddings = np.load(os.path.join(data_path, "job_embeddings.npy"))
        else:
            self._compute_embeddings()

    def _compute_embeddings(self):
        resume_texts = self.resumes['text'].fillna('').tolist()
        job_texts = self.jobs['text'].fillna('').tolist()
        self.resume_embeddings = self.model.encode(resume_texts, convert_to_numpy=True)
        self.job_embeddings = self.model.encode(job_texts, convert_to_numpy=True)

    def _cosine(self, a, b):
        return cosine_similarity(a.reshape(1, -1), b).flatten()

    def extract_skills(self, text):
        doc = self.nlp(text)
        return [ent.text for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT", "WORK_OF_ART", "SKILL"]]

    def match_job_against_resumes(self, job_text, top_k=5):
        job_emb = self.model.encode([job_text], convert_to_numpy=True)[0]
        sims = self._cosine(job_emb, self.resume_embeddings)
        idx = sims.argsort()[::-1][:top_k]
        results = []
        job_skills = self.extract_skills(job_text)
        for i in idx:
            row = self.resumes.iloc[i]
            resume_skills = self.extract_skills(row['text'])
            matched_skills = list(set(job_skills) & set(resume_skills))
            results.append({
                "title": row.get("title", "Unnamed"),
                "score": float(sims[i]),
                "excerpt": row['text'][:300] + "...",
                "skills": matched_skills
            })
        return {"query": job_text[:200], "results": results}

    def match_resume_against_jobs(self, resume_text, top_k=5):
        resume_emb = self.model.encode([resume_text], convert_to_numpy=True)[0]
        sims = self._cosine(resume_emb, self.job_embeddings)
        idx = sims.argsort()[::-1][:top_k]
        results = []
        resume_skills = self.extract_skills(resume_text)
        for i in idx:
            row = self.jobs.iloc[i]
            job_skills = self.extract_skills(row['text'])
            matched_skills = list(set(resume_skills) & set(job_skills))
            results.append({
                "title": row.get("title", "Unnamed"),
                "score": float(sims[i]),
                "excerpt": row['text'][:300] + "...",
                "skills": matched_skills
            })
        return {"query": resume_text[:200], "results": results}
