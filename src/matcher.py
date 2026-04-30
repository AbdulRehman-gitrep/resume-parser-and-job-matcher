import math

import numpy as np


def compute_similarity(resume_text, job_text):
    if not resume_text or not job_text:
        return 0.0

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform([resume_text, job_text])
        score = cosine_similarity(tfidf[0], tfidf[1])[0][0]
        return float(score)
    except Exception:
        return _numpy_tfidf_cosine(resume_text, job_text)


def _numpy_tfidf_cosine(text_a, text_b):
    tokens_a = text_a.split()
    tokens_b = text_b.split()
    vocab = sorted(set(tokens_a) | set(tokens_b))
    if not vocab:
        return 0.0

    tf_a = _term_frequency(tokens_a, vocab)
    tf_b = _term_frequency(tokens_b, vocab)

    doc_count = np.array([
        int(token in tokens_a) + int(token in tokens_b) for token in vocab
    ])
    idf = np.log((1 + 2) / (1 + doc_count)) + 1

    tfidf_a = tf_a * idf
    tfidf_b = tf_b * idf

    denom = (np.linalg.norm(tfidf_a) * np.linalg.norm(tfidf_b))
    if denom == 0:
        return 0.0
    return float(np.dot(tfidf_a, tfidf_b) / denom)


def _term_frequency(tokens, vocab):
    counts = {token: 0 for token in vocab}
    for token in tokens:
        if token in counts:
            counts[token] += 1
    max_count = max(counts.values()) if counts else 1
    return np.array([counts[token] / max_count for token in vocab], dtype=float)
