"""
parser_improved.py
------------------
Addresses Limitation #3: en_core_web_sm is weak on domain-specific entities.

Improvements:
  1. Upgrades to en_core_web_lg when available (better accuracy, same API)
  2. Adds an EntityRuler with hand-crafted patterns for job titles, tech skills,
     and degree types
  3. Falls back gracefully: lg -> md -> sm, in that order
"""

from typing import Dict, List

import spacy
from spacy.language import Language
from spacy.tokens import Doc


def _load_best_model() -> Language:
    """Try to load the largest available English spaCy model."""
    for model_name in ("en_core_web_lg", "en_core_web_md", "en_core_web_sm"):
        try:
            nlp = spacy.load(model_name)
            print(f"[parser] Loaded model: {model_name}")
            return nlp
        except OSError:
            continue
    raise RuntimeError(
        "No spaCy English model found. Run: python -m spacy download en_core_web_sm"
    )


_NLP = _load_best_model()

if "entity_ruler" not in _NLP.pipe_names:
    ruler = _NLP.add_pipe("entity_ruler", before="ner")
else:
    ruler = _NLP.get_pipe("entity_ruler")

DOMAIN_PATTERNS = [
    {"label": "JOB_TITLE", "pattern": [{"LOWER": "software"}, {"LOWER": "engineer"}]},
    {"label": "JOB_TITLE", "pattern": [{"LOWER": "data"}, {"LOWER": "scientist"}]},
    {"label": "JOB_TITLE", "pattern": [{"LOWER": "data"}, {"LOWER": "engineer"}]},
    {
        "label": "JOB_TITLE",
        "pattern": [{"LOWER": "machine"}, {"LOWER": "learning"}, {"LOWER": "engineer"}],
    },
    {"label": "JOB_TITLE", "pattern": [{"LOWER": "ml"}, {"LOWER": "engineer"}]},
    {"label": "JOB_TITLE", "pattern": [{"LOWER": "nlp"}, {"LOWER": "engineer"}]},
    {"label": "JOB_TITLE", "pattern": [{"LOWER": "devops"}, {"LOWER": "engineer"}]},
    {"label": "JOB_TITLE", "pattern": [{"LOWER": "full"}, {"LOWER": "stack"}, {"LOWER": "developer"}]},
    {"label": "JOB_TITLE", "pattern": [{"LOWER": "backend"}, {"LOWER": "developer"}]},
    {"label": "JOB_TITLE", "pattern": [{"LOWER": "frontend"}, {"LOWER": "developer"}]},
    {"label": "JOB_TITLE", "pattern": [{"LOWER": "product"}, {"LOWER": "manager"}]},
    {
        "label": "TECH_SKILL",
        "pattern": [
            {
                "LOWER": {
                    "IN": [
                        "python",
                        "java",
                        "javascript",
                        "typescript",
                        "kotlin",
                        "swift",
                        "go",
                        "rust",
                        "scala",
                        "r",
                        "matlab",
                        "tensorflow",
                        "pytorch",
                        "keras",
                        "sklearn",
                        "scikit-learn",
                        "flask",
                        "django",
                        "fastapi",
                        "react",
                        "vue",
                        "angular",
                        "docker",
                        "kubernetes",
                        "git",
                        "jenkins",
                        "terraform",
                        "aws",
                        "azure",
                        "gcp",
                        "spark",
                        "hadoop",
                        "postgresql",
                        "mysql",
                        "mongodb",
                        "redis",
                    ]
                }
            }
        ],
    },
    {"label": "TECH_SKILL", "pattern": [{"LOWER": "machine"}, {"LOWER": "learning"}]},
    {"label": "TECH_SKILL", "pattern": [{"LOWER": "deep"}, {"LOWER": "learning"}]},
    {
        "label": "TECH_SKILL",
        "pattern": [
            {"LOWER": "natural"},
            {"LOWER": "language"},
            {"LOWER": "processing"},
        ],
    },
    {"label": "TECH_SKILL", "pattern": [{"LOWER": "computer"}, {"LOWER": "vision"}]},
    {
        "label": "TECH_SKILL",
        "pattern": [{"LOWER": "large"}, {"LOWER": "language"}, {"LOWER": "model"}],
    },
    {"label": "TECH_SKILL", "pattern": [{"LOWER": "power"}, {"LOWER": "bi"}]},
    {"label": "TECH_SKILL", "pattern": [{"LOWER": "scikit"}, {"LOWER": "-"}, {"LOWER": "learn"}]},
    {"label": "TECH_SKILL", "pattern": [{"LOWER": "node"}, {"LOWER": "."}, {"LOWER": "js"}]},
    {"label": "DEGREE", "pattern": [{"LOWER": {"IN": ["b.sc.", "bsc", "b.s.", "bachelor's", "bachelor"]}}]},
    {"label": "DEGREE", "pattern": [{"LOWER": {"IN": ["m.sc.", "msc", "m.s.", "master's", "master"]}}]},
    {"label": "DEGREE", "pattern": [{"LOWER": {"IN": ["phd", "ph.d.", "doctorate", "doctoral"]}}]},
    {
        "label": "DEGREE",
        "pattern": [
            {"LOWER": "bachelor"},
            {"LOWER": "of"},
            {"LOWER": {"IN": ["science", "engineering", "arts", "technology"]}},
        ],
    },
    {
        "label": "DEGREE",
        "pattern": [
            {"LOWER": "master"},
            {"LOWER": "of"},
            {"LOWER": {"IN": ["science", "engineering", "arts", "technology"]}},
        ],
    },
]

ruler.add_patterns(DOMAIN_PATTERNS)


def parse_resume(text: str) -> Dict[str, any]:
    """
    Run NER on resume text and return structured entities.
    """
    doc: Doc = _NLP(text)

    name = None
    organizations: List[str] = []
    dates: List[str] = []
    job_titles: List[str] = []
    tech_skills: List[str] = []
    degrees: List[str] = []
    all_entities = []

    seen = set()

    for ent in doc.ents:
        key = (ent.text.strip().lower(), ent.label_)
        if key in seen:
            continue
        seen.add(key)

        label = ent.label_
        value = ent.text.strip()
        all_entities.append((value, label))

        if label == "PERSON" and name is None:
            name = value
        elif label == "ORG":
            organizations.append(value)
        elif label in ("DATE", "TIME"):
            dates.append(value)
        elif label == "JOB_TITLE":
            job_titles.append(value)
        elif label == "TECH_SKILL":
            tech_skills.append(value)
        elif label == "DEGREE":
            degrees.append(value)

    return {
        "name": name,
        "organizations": organizations,
        "dates": dates,
        "job_titles": job_titles,
        "tech_skills": tech_skills,
        "degrees": degrees,
        "all_entities": all_entities,
    }


def parse_entities(text: str) -> Dict[str, any]:
    """Backwards-compatible wrapper used by the app."""
    if not text:
        return {
            "name": None,
            "organizations": [],
            "dates": [],
            "job_titles": [],
            "tech_skills": [],
            "degrees": [],
            "all_entities": [],
        }
    return parse_resume(text)


if __name__ == "__main__":
    sample = """
    John Smith
    Software Engineer | ML Engineer
    B.Sc. Computer Science, MIT, 2020 - 2024

    Experience:
    Data Scientist at Google (June 2024 - Present)
    - Built NLP pipelines using Python and TensorFlow
    - Deployed Docker containers on AWS
    - Applied deep learning for computer vision tasks

    Education:
    Master of Science in Artificial Intelligence, Stanford University, 2022
    """

    result = parse_resume(sample)
    print("Name         :", result["name"])
    print("Orgs         :", result["organizations"])
    print("Dates        :", result["dates"])
    print("Job Titles   :", result["job_titles"])
    print("Tech Skills  :", result["tech_skills"])
    print("Degrees      :", result["degrees"])
