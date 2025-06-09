from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

app = FastAPI()

# Load recruiter job data from CSV
job_df = pd.read_csv("recruiter_jobs.csv")

# Convert comma-separated skill strings into lists
job_df["skills"] = job_df["skills"].apply(lambda x: [skill.strip() for skill in str(x).split(',')])

class MatchRequest(BaseModel):
    student_id: int
    selected_role: str
    student_scores: Dict[str, float]

@app.post("/api/match-skills")
def match_jobs(request: MatchRequest):
    student_scores = request.student_scores
    if not student_scores:
        return {"status": "error", "message": "Student scores are missing."}

    # Prepare all skill columns from student and all jobs
    all_skills = set(student_scores.keys())
    for skills in job_df["skills"]:
        all_skills.update(skills)
    all_skills = sorted(all_skills)

    # Student vector
    student_vec = np.array([student_scores.get(skill, 0) for skill in all_skills])

    results = []

    for _, row in job_df.iterrows():
        job_vec = np.array([1 if skill in row["skills"] else 0 for skill in all_skills])

        # Optional: Replace binary vector with expected skill score values (out of 20) if you have them
        similarity = cosine_similarity([student_vec], [job_vec])[0][0]

        results.append({
            "job_title": row["job_title"],
            "recruiter_name": row["recruiter_name"],
            "company_name": row["company_name"],
            "job_description": row["job_description"],
            "skills": row["skills"],
            "match_score": round(float(similarity * 100), 2)
        })

    results.sort(key=lambda x: x["match_score"], reverse=True)

    return {
        "student_id": request.student_id,
        "best_matches": results,
        "status": "success"
    }
