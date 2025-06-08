from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

class Job(BaseModel):
    job_title: str
    skills: dict

class MatchRequest(BaseModel):
    student_id: int
    student_scores: dict
    job_list: list[Job]

@app.post("/match-jobs")
def match_jobs(request: MatchRequest):
    student_scores = request.student_scores
    job_list = request.job_list

    if not student_scores or not job_list:
        return {"status": "error", "message": "Missing student scores or job requirements."}

    # Combine all skills
    all_skills = set(student_scores.keys())
    for job in job_list:
        all_skills.update(job.skills.keys())
    all_skills = sorted(all_skills)

    # Convert student scores to vector
    student_vec = np.array([student_scores.get(skill, 0) for skill in all_skills])

    results = []
    for job in job_list:
        job_vec = np.array([job.skills.get(skill, 0) for skill in all_skills])
        similarity = cosine_similarity([student_vec], [job_vec])[0][0]
        results.append({
            "job_title": job.job_title,
            "match_score": round(float(similarity * 100), 2)
        })

    results.sort(key=lambda x: x["match_score"], reverse=True)

    return {
        "student_id": request.student_id,
        "best_matches": results,
        "status": "success"
    }
