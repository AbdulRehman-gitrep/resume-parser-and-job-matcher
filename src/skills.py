import re


VARIANT_MAP = {
    "node.js": "nodejs",
    "node js": "nodejs",
    "express.js": "expressjs",
    "express js": "expressjs",
    "c++": "cpp",
    "c#": "csharp",
    "ci/cd": "cicd",
    "ci cd": "cicd",
    "restful": "rest",
}


def load_skills(path):
    skills_by_category = {}
    current_category = None

    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#"):
                continue
            if line.startswith("[") and line.endswith("]"):
                current_category = line[1:-1].strip()
                skills_by_category[current_category] = []
                continue
            if current_category is None:
                current_category = "General"
                skills_by_category.setdefault(current_category, [])
            skills_by_category[current_category].append(line)

    return skills_by_category


def extract_skills(text, skills_list):
    normalized_text = normalize_for_matching(text)
    matches = []

    for skill in skills_list:
        normalized_skill = normalize_for_matching(skill)
        if not normalized_skill:
            continue
        if _has_skill(normalized_text, normalized_skill):
            matches.append(skill)

    return sorted(set(matches))


def extract_skills_by_category(text, skills_by_category):
    results = {}
    for category, skills in skills_by_category.items():
        results[category] = extract_skills(text, skills)
    return results


def flatten_skills(skills_by_category):
    flat = []
    for skills in skills_by_category.values():
        flat.extend(skills)
    return sorted(set(flat))


def normalize_for_matching(text):
    if not text:
        return ""
    normalized = text.lower()
    for key, value in VARIANT_MAP.items():
        normalized = normalized.replace(key, value)
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _has_skill(normalized_text, normalized_skill):
    escaped = re.escape(normalized_skill)
    pattern = r"\b" + escaped.replace(" ", r"\\s+") + r"\b"
    return re.search(pattern, normalized_text) is not None
