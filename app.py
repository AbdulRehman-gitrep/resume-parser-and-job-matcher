import os

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from src.cleaner import clean_text
from src.extractor import extract_text
from src.matcher import compute_similarity
from src.parser import parse_entities
from src.scorer import calculate_score
from src.skills import extract_skills_by_category, flatten_skills, load_skills


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
SKILLS_PATH = os.path.join(BASE_DIR, "data", "skills.txt")
ALLOWED_EXTENSIONS = {".pdf", ".docx"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files.get("resume")
    job_description = request.form.get("job_description", "").strip()

    if not file or file.filename == "":
        return render_template("index.html", error="Please upload a resume file.")
    if not job_description:
        return render_template("index.html", error="Please provide a job description.")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return render_template(
            "index.html",
            error="Unsupported file type. Please upload a PDF or DOCX.",
        )

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    file.save(save_path)

    try:
        resume_text = extract_text(save_path)
    except Exception as exc:
        return render_template(
            "index.html",
            error=f"Failed to extract resume text: {exc}",
        )

    cleaned_resume = clean_text(resume_text)
    cleaned_job = clean_text(job_description)

    try:
        entities = parse_entities(resume_text)
    except Exception as exc:
        return render_template(
            "index.html",
            error=str(exc),
        )

    skills_by_category = load_skills(SKILLS_PATH)
    resume_skills_by_category = extract_skills_by_category(cleaned_resume, skills_by_category)
    job_skills_by_category = extract_skills_by_category(cleaned_job, skills_by_category)

    resume_skills = flatten_skills(resume_skills_by_category)
    job_skills = flatten_skills(job_skills_by_category)

    matched_skills = sorted(set(resume_skills) & set(job_skills))
    missing_skills = sorted(set(job_skills) - set(resume_skills))

    similarity = compute_similarity(cleaned_resume, cleaned_job)
    score_data = calculate_score(similarity, matched_skills, len(job_skills))

    similarity_pct = round(similarity * 100, 2)
    similarity_weighted = similarity * 0.6
    similarity_weighted_pct = round(similarity_weighted * 100, 2)

    skill_ratio = score_data["skill_ratio"]
    skill_ratio_pct = round(skill_ratio * 100, 2)
    skill_weighted = skill_ratio * 0.4
    skill_weighted_pct = round(skill_weighted * 100, 2)

    final_score = score_data["final_score"]
    final_score_pct = round(final_score * 100, 2)

    final_score_formula = (
        f"(0.6 x {similarity_pct}%) + (0.4 x {skill_ratio_pct}%) = {final_score_pct}%"
    )

    category_breakdown = []
    for category in skills_by_category:
        job_cat = job_skills_by_category.get(category, [])
        resume_cat = resume_skills_by_category.get(category, [])
        matched_cat = sorted(set(job_cat) & set(resume_cat))
        missing_cat = sorted(set(job_cat) - set(resume_cat))
        if not job_cat:
            continue
        category_breakdown.append(
            {
                "category": category,
                "matched": matched_cat,
                "missing": missing_cat,
            }
        )

    return render_template(
        "result.html",
        resume_name=entities.get("name"),
        organizations=entities.get("organizations"),
        dates=entities.get("dates"),
        similarity_pct=similarity_pct,
        similarity_weighted_pct=similarity_weighted_pct,
        skill_ratio_pct=skill_ratio_pct,
        skill_weighted_pct=skill_weighted_pct,
        final_score_pct=final_score_pct,
        final_score_formula=final_score_formula,
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        resume_skills=resume_skills,
        job_skills=job_skills,
        category_breakdown=category_breakdown,
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
