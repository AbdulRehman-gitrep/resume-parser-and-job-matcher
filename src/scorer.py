def calculate_score(similarity, matched_skills, total_job_skills):
    matched_count = len(set(matched_skills))
    skill_ratio = matched_count / total_job_skills if total_job_skills else 0.0
    final_score = (0.6 * similarity) + (0.4 * skill_ratio)
    return {"final_score": final_score, "skill_ratio": skill_ratio}
