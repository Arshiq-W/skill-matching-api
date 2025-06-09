from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import re

app = FastAPI()

# Set your recruiter file name (CSV or Excel)
CSV_FILE = "job_data.csv"  # or change to "job_data.xlsx" if Excel

class StudentRequest(BaseModel):
    student_id: str
    student_scores: Dict[str, float]

# Parse scores: handles numbers and ranges like "12-15"
def parse_score(cell):
    if pd.isna(cell):
        return 0
    if isinstance(cell, (int, float)):
        return float(cell)
    match = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", str(cell))
    if match:
        low, high = map(float, match.groups())
        return (low + high) / 2
    try:
        return float(cell)
    except:
        return 0

# Load and format job data from file
def load_job_data():
    ext = os.path.splitext(CSV_FILE)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(CSV_FILE)
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(CSV_FILE)
    else:
        raise ValueError("Unsupported file type")

    job_list = []
    for _, row in df.iterrows():
        job_title = row["job_title"]
        skills = {col: parse_score(row[col]) for col in df.columns if col != "job_title"}
        job_list.append({"job_title": job_title, "skills": skills})
    return job_list

@app.post("/api/match-skills")
def match_skills(request: StudentRequest):
    student_scores = request.student_scores
    job_list = load_job_data()

    if not student_scores or not job_list:
        return {"status": "error", "message": "Missing student scores or job data."}

    # Align all skills
    all_skills = set(student_scores.keys())
    for job in job_list:
        all_skills.update(job["skills"].keys())
    all_skills = sorted(all_skills)

    # Vector for student
    student_vec = np.array([student_scores.get(skill, 0) for skill in all_skills])

    results = []
    for job in job_list:
        job_vec = np.array([job["skills"].get(skill, 0) for skill in all_skills])
        similarity = cosine_similarity([student_vec], [job_vec])[0][0]
        results.append({
            "job_title": job["job_title"],
            "match_score": round(float(similarity * 100), 2)
        })

    results.sort(key=lambda x: x["match_score"], reverse=True)

    return {
        "student_id": request.student_id,
        "best_matches": results,
        "status": "success"
    }
