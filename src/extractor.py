import os

import pdfplumber
from docx import Document


SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}")

    if ext == ".pdf":
        return _extract_pdf_text(file_path)

    return _extract_docx_text(file_path)


def _extract_pdf_text(file_path):
    text_parts = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts).strip()


def _extract_docx_text(file_path):
    doc = Document(file_path)
    text_parts = [para.text for para in doc.paragraphs if para.text]
    return "\n".join(text_parts).strip()
