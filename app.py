import os

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from src.extractor import extract_text
from src.matcher import compute_similarity
from src.parser import parse_entities
from src.scorer import calculate_score
from src.skills import extract_skills


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
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

    try:
        entities = parse_entities(resume_text)
    except Exception as exc:
        return render_template(
            "index.html",
            error=str(exc),
        )

    resume_skill_result = extract_skills(resume_text)
    job_skill_result = extract_skills(job_description)

    resume_skills = sorted({skill for skill, _ in resume_skill_result["matched"]})
    job_skills = sorted({skill for skill, _ in job_skill_result["matched"]})

    matched_skills = sorted(set(resume_skills) & set(job_skills))
    missing_skills = sorted(set(job_skills) - set(resume_skills))

    similarity_result = compute_similarity(resume_text, job_description)
    similarity_score = similarity_result["score"]
    score_data = calculate_score(similarity_score, matched_skills, len(job_skills))

    similarity_pct = similarity_result["percentage"]
    similarity_method = similarity_result["method"]
    tfidf_pct = similarity_result["tfidf_percentage"]

    similarity_weighted = similarity_score * 0.6
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
    for category, job_cat in job_skill_result["by_category"].items():
        resume_cat = resume_skill_result["by_category"].get(category, [])
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
        similarity_method=similarity_method,
        tfidf_pct=tfidf_pct,
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
        job_titles=entities.get("job_titles"),
        tech_skills=entities.get("tech_skills"),
        degrees=entities.get("degrees"),
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
