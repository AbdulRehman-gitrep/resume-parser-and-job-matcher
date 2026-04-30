"""
matcher_improved.py
-------------------
Addresses Limitation #1: TF-IDF misses semantic meaning.

Improvement: Uses sentence-transformers (all-MiniLM-L6-v2) for semantic
similarity. Falls back to TF-IDF cosine similarity if the library is
unavailable, preserving the original behavior.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Attempt to load sentence-transformers ---
try:
    from sentence_transformers import SentenceTransformer, util as st_util

    _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    _USE_SEMANTIC = True
    print("[matcher] Using semantic similarity (sentence-transformers).")
except ImportError:
    _MODEL = None
    _USE_SEMANTIC = False
    print("[matcher] sentence-transformers not found. Falling back to TF-IDF.")


def semantic_similarity(text1: str, text2: str) -> float:
    """
    Compute semantic similarity using sentence embeddings.
    Returns a float in [0, 1].
    """
    emb1, emb2 = _MODEL.encode([text1, text2], convert_to_tensor=True)
    score = st_util.cos_sim(emb1, emb2).item()
    return round(max(0.0, min(1.0, score)), 4)


def tfidf_similarity(text1: str, text2: str) -> float:
    """
    Compute TF-IDF cosine similarity (original approach).
    Returns a float in [0, 1].
    """
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return round(float(score), 4)
    except Exception:
        return 0.0


def compute_similarity(resume_text: str, job_text: str) -> dict:
    """
    Main entry point.

    Returns a dict with:
        - method       : 'semantic' | 'tfidf'
        - score        : float [0, 1]
        - percentage   : float [0, 100]
        - tfidf_score  : float (always computed for comparison)
    """
    if not resume_text or not job_text:
        return {
            "method": "tfidf",
            "score": 0.0,
            "percentage": 0.0,
            "tfidf_score": 0.0,
            "tfidf_percentage": 0.0,
        }

    tfidf_score = tfidf_similarity(resume_text, job_text)

    if _USE_SEMANTIC:
        sem_score = semantic_similarity(resume_text, job_text)
        return {
            "method": "semantic",
            "score": sem_score,
            "percentage": round(sem_score * 100, 2),
            "tfidf_score": tfidf_score,
            "tfidf_percentage": round(tfidf_score * 100, 2),
        }

    return {
        "method": "tfidf",
        "score": tfidf_score,
        "percentage": round(tfidf_score * 100, 2),
        "tfidf_score": tfidf_score,
        "tfidf_percentage": round(tfidf_score * 100, 2),
    }


if __name__ == "__main__":
    resume = "Experienced Python developer with skills in machine learning and data analysis."
    job = "Looking for an ML engineer proficient in Python and statistical modelling."

    result = compute_similarity(resume, job)
    print(f"Method     : {result['method']}")
    print(f"Score      : {result['score']}")
    print(f"Percentage : {result['percentage']}%")
    print(f"TF-IDF     : {result['tfidf_percentage']}%")
