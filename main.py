from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(
    title="Career.Connect Job Matching API",
    description="API for matching students with jobs based on their quiz scores and skills",
    version="2.0.0"
)

# Job skill structure from Laravel payload
class JobSkill(BaseModel):
    skill_name: str
    experience_level: str
    is_required: bool

# Job data structure from Laravel payload
class JobData(BaseModel):
    job_id: int
    job_title: str
    company_name: str
    job_type: str
    experience_level: str
    location: str
    salary_range: Optional[str]
    job_description: str
    application_deadline: Optional[str]
    recruiter_name: str
    recruiter_email: str
    domain_name: str
    required_skills: List[JobSkill]
    matching_skills: List[str]

# Request structure from Laravel
class MatchRequest(BaseModel):
    student_id: str  # Changed to string to match Laravel payload
    selected_role: str
    student_scores: Dict[str, float]
    available_jobs: List[JobData]  # Jobs sent from Laravel database

@app.get("/")
def read_root():
    return {
        "message": "Career.Connect Job Matching API v2.0",
        "status": "active",
        "endpoints": {
            "match_skills": "/api/match-skills",
            "health": "/health"
        }
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "version": "2.0.0"}

@app.post("/api/match-skills")
def match_jobs(request: MatchRequest):
    try:
        student_scores = request.student_scores
        available_jobs = request.available_jobs

        # Validation
        if not student_scores:
            return {"status": "error", "message": "Student scores are missing."}

        if not available_jobs:
            return {"status": "error", "message": "No jobs available for matching."}

        # Prepare all skill columns from student and job data sent in payload
        all_skills = set(student_scores.keys())

        # Extract skills from job data received in payload
        for job in available_jobs:
            job_skills = [skill.skill_name for skill in job.required_skills]
            all_skills.update(job_skills)
            all_skills.update(job.matching_skills)

        all_skills = sorted(all_skills)

        # Student vector - normalize scores to 0-1 range (assuming max score is 20)
        max_score = 20.0
        student_vec = np.array([min(student_scores.get(skill, 0) / max_score, 1.0) for skill in all_skills])

        results = []

        # Process each job sent in the payload
        for job in available_jobs:
            # Create job skill vector with weighted importance based on job requirements
            job_skills_dict = {}
            for skill in job.required_skills:
                skill_name = skill.skill_name
                experience_level = skill.experience_level
                is_required = skill.is_required

                # Weight skills based on experience level and requirement
                weight = 1.0
                if experience_level == "expert":
                    weight = 1.0
                elif experience_level == "intermediate":
                    weight = 0.8
                elif experience_level == "beginner":
                    weight = 0.6

                # Boost required skills
                if is_required:
                    weight *= 1.2

                job_skills_dict[skill_name] = weight

            # Create job vector based on skills sent in payload
            job_vec = np.array([job_skills_dict.get(skill, 0) for skill in all_skills])

            # Calculate cosine similarity between student and job
            if np.linalg.norm(student_vec) == 0 or np.linalg.norm(job_vec) == 0:
                similarity = 0
            else:
                similarity = cosine_similarity([student_vec], [job_vec])[0][0]

            # Calculate bonus for matching skills (skills student has completed quizzes for)
            matching_bonus = 0
            total_required_skills = len([s for s in job.required_skills if s.is_required])
            if total_required_skills > 0:
                matching_required_skills = len([s for s in job.matching_skills
                                              if any(rs.skill_name == s and rs.is_required
                                                   for rs in job.required_skills)])
                matching_bonus = (matching_required_skills / total_required_skills) * 0.3

            # Final score with bonus
            final_score = min((similarity + matching_bonus) * 100, 100)

            # Prepare result with all job data received from payload
            results.append({
                "job_id": job.job_id,
                "job_title": job.job_title,
                "recruiter_name": job.recruiter_name,
                "company_name": job.company_name,
                "job_type": job.job_type,
                "experience_level": job.experience_level,
                "location": job.location,
                "salary_range": job.salary_range,
                "job_description": job.job_description,
                "application_deadline": job.application_deadline,
                "recruiter_email": job.recruiter_email,
                "domain_name": job.domain_name,
                "skills": [skill.skill_name for skill in job.required_skills],
                "required_skills_detailed": [
                    {
                        "skill_name": skill.skill_name,
                        "experience_level": skill.experience_level,
                        "is_required": skill.is_required
                    } for skill in job.required_skills
                ],
                "matching_skills": job.matching_skills,
                "match_score": round(float(final_score), 2)
            })

        # Sort by match score (highest first)
        results.sort(key=lambda x: x["match_score"], reverse=True)

        return {
            "student_id": request.student_id,
            "selected_role": request.selected_role,
            "best_matches": results,
            "total_jobs_analyzed": len(available_jobs),
            "status": "success"
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Error processing job matching: {str(e)}",
            "student_id": request.student_id if hasattr(request, 'student_id') else "unknown"
        }
