"""
app.py — FastAPI Server for ExamGuard Study Buddy & Exam Coach
==============================================================
Architecture:
  Study mode  → cactus built-in RAG (cactus_init with corpus_dir + cactus_rag_query)
                No hardcoded prompts — retrieval is fully automatic.
  Practice    → FunctionGemma tool-calling for structured score output
  STT         → Whisper-tiny via cactus_transcribe

Endpoints:
  GET  /            → mobile UI
  POST /upload-pdf  → extract PDF text → write corpus.txt → reload RAG model
  POST /ask         → cactus_rag_query(question) → top-k chunks returned
  POST /quiz        → pick sentences from corpus as practice question
  POST /feedback    → FunctionGemma scores the student's answer
  POST /transcribe  → Whisper STT
  GET  /health
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from stt import SpeechToText
from coach import ExamCoach
from pdf_reader import extract_text, get_relevant_context

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="ExamGuard")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Models ───────────────────────────────────────────────────────────────────
print("Loading on-device models…")
try:
    stt = SpeechToText()
    print("  ✓ Whisper (STT) ready")
except RuntimeError as e:
    stt = None
    print(f"  ✗ Whisper not available: {e}")

coach = ExamCoach()
print("  ✓ FunctionGemma (scoring) ready")

_pdf_text = ""
_pdf_name = ""

TEMPLATE_PATH = Path(__file__).parent / "templates" / "index.html"
print("\nExamGuard is running at http://localhost:8000\n")


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    if not TEMPLATE_PATH.exists():
        return HTMLResponse("<h1>UI not found</h1>")
    return HTMLResponse(content=TEMPLATE_PATH.read_text())


@app.get("/health")
async def health():
    return {
        "status":     "ok",
        "stt_ready":  stt is not None,
        "pdf_loaded": bool(_pdf_text),
        "pdf_name":   _pdf_name or None,
        "on_device":  True,
    }


@app.post("/upload-pdf")
async def upload_pdf(pdf: UploadFile = File(...)):
    """Extract PDF text and store in memory for keyword-based retrieval."""
    global _pdf_text, _pdf_name

    if not pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(await pdf.read())
        tmp_pdf = tmp.name

    try:
        text = extract_text(tmp_pdf)
        if not text.strip():
            raise HTTPException(status_code=400, detail="PDF appears to be empty or image-only.")
        _pdf_text = text
        _pdf_name = pdf.filename
        return JSONResponse({"success": True, "filename": pdf.filename, "chars": len(text)})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.unlink(tmp_pdf)
        except OSError:
            pass


class AskRequest(BaseModel):
    question: str


@app.post("/ask")
async def ask(req: AskRequest):
    """
    Study mode: find the most relevant passages in the PDF for the question.
    Uses keyword + density scoring across paragraphs — returns text directly.
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Please ask a question.")

    if not _pdf_text:
        return JSONResponse({
            "success":    True,
            "answer":     "Please upload your study PDF first using the button at the top.",
            "key_points": [],
        })

    # Get top relevant passage (main answer)
    answer = get_relevant_context(_pdf_text, req.question, max_chars=800)

    # Get key points: search again with slightly different scope for variety
    import re
    sentences = re.split(r'(?<=[.!?])\s+', answer)
    key_points = [s.strip() for s in sentences if len(s.strip()) > 40]

    # Main explanation = first 600 chars; rest become key points
    main = answer[:600].strip()
    bullets = key_points[2:5] if len(key_points) > 2 else []

    return JSONResponse({
        "success":    True,
        "answer":     main,
        "key_points": bullets,
    })


class QuizRequest(BaseModel):
    topic: str = ""


@app.post("/quiz")
async def quiz(req: QuizRequest):
    """Practice mode: pick a sentence from the PDF as a practice question."""
    import re

    if _pdf_text:
        query = req.topic.strip() or "definition concept example"
        passage = get_relevant_context(_pdf_text, query, max_chars=600)
        sentences = re.split(r'(?<=[.!?])\s+', passage)
        long_s = [s.strip() for s in sentences if len(s.strip()) > 50]
        if long_s:
            return JSONResponse({
                "success":  True,
                "question": f"Explain in your own words: \"{long_s[0][:150]}\"",
                "hint":     long_s[1][:100] if len(long_s) > 1 else "Think about the key terms.",
            })

    return JSONResponse({
        "success":  True,
        "question": "Explain the main topic covered in your study material.",
        "hint":     "Use specific terms and examples from the text.",
    })


class FeedbackRequest(BaseModel):
    transcript: str
    question:   str = ""


@app.post("/feedback")
async def feedback(req: FeedbackRequest):
    """
    Practice mode: FunctionGemma scores the student's answer via tool-calling.
    Uses RAG to optionally pull context for scoring.
    """
    pdf_context = ""
    if _pdf_text and (req.question or req.transcript):
        query = req.question or req.transcript
        pdf_context = get_relevant_context(_pdf_text, query, max_chars=400)

    try:
        result = coach.get_feedback(
            transcript=req.transcript,
            question=req.question,
            pdf_context=pdf_context,
        )
        return JSONResponse({
            "score":    result["score"],
            "bullets":  result["bullets"],
            "used_pdf": bool(pdf_context),
            "success":  True,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    if stt is None:
        raise HTTPException(status_code=503, detail="Whisper model not loaded.")

    suffix = ".wav"
    ct = audio.content_type or ""
    if "mp4" in ct:   suffix = ".mp4"
    elif "webm" in ct: suffix = ".webm"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        transcript = stt.transcribe(tmp_path)
        return JSONResponse({"transcript": transcript, "success": True})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
