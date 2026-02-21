"""
pdf_reader.py — On-device PDF text extraction and search
=========================================================
Extracts text from uploaded study PDFs and finds the most
relevant passages for a given exam question.
No cloud, no external APIs — pure local processing.
"""

import re
from pathlib import Path


def extract_text(pdf_path: str) -> str:
    """
    Extract all text from a PDF file.
    Returns plain text string.
    """
    try:
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text.strip())
        return "\n\n".join(pages)
    except ImportError:
        raise RuntimeError("pypdf not installed. Run: pip install pypdf")
    except Exception as e:
        raise RuntimeError(f"Could not read PDF: {e}")


def get_relevant_context(full_text: str, question: str, max_chars: int = 1500) -> str:
    """
    Find the most relevant section of the PDF for a given question.

    Strategy:
    1. Split PDF into paragraphs
    2. Score each paragraph by keyword overlap with the question
    3. Return top-scoring paragraphs up to max_chars
    """
    if not full_text.strip():
        return ""

    if not question.strip():
        # No question — return the beginning of the document
        return full_text[:max_chars].strip()

    # Extract keywords from question (ignore stop words)
    stop_words = {
        "a","an","the","is","are","was","were","be","been","being",
        "have","has","had","do","does","did","will","would","could",
        "should","may","might","shall","what","when","where","who",
        "which","how","why","and","or","but","in","on","at","to",
        "for","of","with","by","from","about","this","that","these",
        "those","my","your","his","her","its","our","their","i","you",
        "he","she","it","we","they","me","him","us","them"
    }
    keywords = set(
        w.lower() for w in re.findall(r'\b\w+\b', question)
        if w.lower() not in stop_words and len(w) > 2
    )

    if not keywords:
        return full_text[:max_chars].strip()

    # Split into paragraphs
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', full_text) if len(p.strip()) > 50]

    if not paragraphs:
        return full_text[:max_chars].strip()

    # Score each paragraph by keyword hits
    def score(para: str) -> int:
        para_lower = para.lower()
        return sum(1 for kw in keywords if kw in para_lower)

    scored = sorted(paragraphs, key=score, reverse=True)

    # Collect top paragraphs up to max_chars
    collected = []
    total = 0
    for para in scored:
        if total + len(para) > max_chars:
            break
        collected.append(para)
        total += len(para)

    return "\n\n".join(collected) if collected else full_text[:max_chars].strip()


# ── Smoke test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pdf_reader.py <file.pdf> [question]")
    else:
        text = extract_text(sys.argv[1])
        question = sys.argv[2] if len(sys.argv) > 2 else ""
        ctx = get_relevant_context(text, question)
        print(f"Extracted {len(text)} chars total")
        print(f"Relevant context ({len(ctx)} chars):\n")
        print(ctx[:500], "...")
