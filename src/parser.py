import spacy


_NLP = None


def _load_model():
    global _NLP
    if _NLP is None:
        try:
            _NLP = spacy.load("en_core_web_sm")
        except OSError as exc:
            raise RuntimeError(
                "spaCy model not found. Install it with: python -m spacy download en_core_web_sm"
            ) from exc
    return _NLP


def parse_entities(text):
    if not text:
        return {"name": None, "organizations": [], "dates": []}

    nlp = _load_model()
    doc = nlp(text)

    name = None
    organizations = []
    dates = []

    for ent in doc.ents:
        if ent.label_ == "PERSON" and name is None:
            name = ent.text
        elif ent.label_ == "ORG":
            organizations.append(ent.text)
        elif ent.label_ == "DATE":
            dates.append(ent.text)

    return {
        "name": name,
        "organizations": sorted(set(organizations)),
        "dates": sorted(set(dates)),
    }
