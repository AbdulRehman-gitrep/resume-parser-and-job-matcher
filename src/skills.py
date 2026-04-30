"""
skills_improved.py
------------------
Addresses Limitation #2: Dictionary skill matching breaks on synonyms/variants.

Improvements:
  1. Synonym map  -- canonical aliases ("ml" -> "machine learning")
  2. Fuzzy match  -- RapidFuzz catches typos like "pytohn" -> "python"
  3. Partial ratio -- handles multi-word skills that appear as substrings
  4. Falls back to pure regex if RapidFuzz is not installed
"""

import re
from typing import Dict, List, Tuple

try:
    from rapidfuzz import fuzz

    _USE_FUZZY = True
    print("[skills] Using fuzzy matching (rapidfuzz).")
except ImportError:
    _USE_FUZZY = False
    print("[skills] rapidfuzz not found. Using regex-only matching.")

SKILL_DICT: Dict[str, List[str]] = {
    "programming_languages": [
        "python",
        "java",
        "javascript",
        "typescript",
        "c++",
        "c#",
        "r",
        "go",
        "kotlin",
        "swift",
        "ruby",
        "php",
        "scala",
        "rust",
    ],
    "ml_ai": [
        "machine learning",
        "deep learning",
        "neural network",
        "nlp",
        "natural language processing",
        "computer vision",
        "reinforcement learning",
        "transfer learning",
        "transformers",
        "llm",
        "generative ai",
    ],
    "data": [
        "data analysis",
        "data science",
        "data engineering",
        "sql",
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "tableau",
        "power bi",
    ],
    "frameworks": [
        "tensorflow",
        "pytorch",
        "scikit-learn",
        "keras",
        "flask",
        "django",
        "fastapi",
        "react",
        "node.js",
        "spring boot",
    ],
    "cloud_devops": [
        "aws",
        "azure",
        "gcp",
        "docker",
        "kubernetes",
        "ci/cd",
        "terraform",
        "ansible",
        "jenkins",
        "git",
    ],
}

ALL_SKILLS: List[Tuple[str, str]] = [
    (skill, category)
    for category, skills in SKILL_DICT.items()
    for skill in skills
]

SYNONYM_MAP: Dict[str, str] = {
    "ml": "machine learning",
    "dl": "deep learning",
    "ai": "machine learning",
    "nlp": "natural language processing",
    "cv": "computer vision",
    "rl": "reinforcement learning",
    "llms": "llm",
    "gen ai": "generative ai",
    "js": "javascript",
    "ts": "typescript",
    "py": "python",
    "cpp": "c++",
    "csharp": "c#",
    "sk-learn": "scikit-learn",
    "sklearn": "scikit-learn",
    "tf": "tensorflow",
    "pt": "pytorch",
    "k8s": "kubernetes",
    "gke": "kubernetes",
    "amazon web services": "aws",
    "microsoft azure": "azure",
    "google cloud": "gcp",
    "google cloud platform": "gcp",
}


def _normalize(text: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation except / and +."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s/+#.]", " ", text)
    return re.sub(r"\s+", " ", text)


def _apply_synonyms(text: str) -> str:
    """Replace known aliases with canonical skill names."""
    for alias, canonical in SYNONYM_MAP.items():
        pattern = r"\b" + re.escape(alias) + r"\b"
        text = re.sub(pattern, canonical, text)
    return text


def _regex_match(text: str, skill: str) -> bool:
    """Exact word-boundary match."""
    pattern = r"\b" + re.escape(skill) + r"\b"
    return bool(re.search(pattern, text))


def _fuzzy_match(text: str, skill: str, threshold: int = 88) -> bool:
    """
    Fuzzy match for single-word skills (handles typos).
    Uses token_set_ratio for multi-word skills (order-insensitive).
    Threshold 88 is deliberately strict to avoid false positives.
    """
    words = text.split()
    if " " not in skill:
        for word in words:
            if fuzz.ratio(word, skill) >= threshold:
                return True
    else:
        skill_words = skill.split()
        n = len(skill_words)
        for i in range(len(words) - n + 1):
            chunk = " ".join(words[i : i + n])
            if fuzz.token_set_ratio(chunk, skill) >= threshold:
                return True
    return False


def extract_skills(text: str) -> Dict[str, any]:
    """
    Extract skills from text with synonym expansion + optional fuzzy matching.

    Returns:
        {
          "matched": [(skill, category), ...],
          "by_category": {category: [skill, ...], ...},
          "method": "fuzzy" | "regex",
        }
    """
    norm = _normalize(text)
    norm = _apply_synonyms(norm)

    matched: List[Tuple[str, str]] = []
    seen: set = set()

    for skill, category in ALL_SKILLS:
        if skill in seen:
            continue

        found = _regex_match(norm, skill)

        if not found and _USE_FUZZY:
            found = _fuzzy_match(norm, skill)

        if found:
            matched.append((skill, category))
            seen.add(skill)

    by_category: Dict[str, List[str]] = {}
    for skill, category in matched:
        by_category.setdefault(category, []).append(skill)

    return {
        "matched": matched,
        "by_category": by_category,
        "method": "fuzzy" if _USE_FUZZY else "regex",
    }


def compare_skills(resume_text: str, job_text: str) -> Dict[str, any]:
    """
    Compare skills between resume and job description.

    Returns matched, missing, extra, and skill coverage ratio.
    """
    resume_result = extract_skills(resume_text)
    job_result = extract_skills(job_text)

    resume_skills = {s for s, _ in resume_result["matched"]}
    job_skills = {s for s, _ in job_result["matched"]}

    matched_skills = resume_skills & job_skills
    missing_skills = job_skills - resume_skills
    extra_skills = resume_skills - job_skills

    ratio = (len(matched_skills) / len(job_skills)) if job_skills else 0.0

    return {
        "matched": sorted(matched_skills),
        "missing": sorted(missing_skills),
        "extra": sorted(extra_skills),
        "ratio": round(ratio, 4),
        "percentage": round(ratio * 100, 2),
    }


if __name__ == "__main__":
    resume = """
        Skilled in Pytohn (typo test) and ML. Experience with sklearn,
        TF, and k8s deployment on GCP. Worked on NLP tasks using transformers.
    """
    job = """
        Requires Python, machine learning, scikit-learn, kubernetes,
        google cloud platform, and natural language processing.
    """
    result = compare_skills(resume, job)
    print("Matched :", result["matched"])
    print("Missing :", result["missing"])
    print("Coverage:", result["percentage"], "%")
