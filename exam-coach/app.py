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
import re
import sys
import tempfile
import shutil
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
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


# ── Exception handler for upload validation (e.g. missing form field "pdf") ─────
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    if request.url.path == "/upload-pdf" and exc.errors():
        msg = "Please select a PDF file and try again. (Form field must be 'pdf'.)"
        return JSONResponse(status_code=422, content={"detail": msg})
    return JSONResponse(status_code=422, content={"detail": exc.errors()})

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
async def upload_pdf(pdf: UploadFile = File(..., description="PDF file (form field name must be 'pdf')")):
    """Extract PDF text and store in memory for keyword-based retrieval."""
    global _pdf_text, _pdf_name

    if not pdf.filename or not str(pdf.filename).lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    content = await pdf.read()
    if not content:
        raise HTTPException(status_code=400, detail="File is empty. Please choose a valid PDF.")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(content)
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

    # Detect greetings / off-topic so we always return a clear response
    _q = req.question.strip().lower()
    _words = set(re.findall(r'\b\w+\b', _q))
    _stop = {"a","an","the","is","are","was","were","be","been","being", "have","has","had","do","does","did","will","would","could", "should","may","might","shall","what","when","where","who","which","how","why","and","or","but","in","on","at","to","for","of","with","by","from","about","this","that","these","those","my","your","his","her","its","our","their","i","you","he","she","it","we","they","me","him","us","them","not","no","s","t","don","isn","can","just","also","very","more","than","hey","hi","hello","sup","yes","no"}
    _meaningful = _words - _stop
    _meaningful = {w for w in _meaningful if len(w) > 2}
    _greetings = {"hey", "hi", "hello", "sup", "whats", "what"}
    if not _meaningful or _meaningful <= _greetings:
        return JSONResponse({
            "success":    True,
            "answer":     "Ask me something about your PDF. For example: \"What is Big Data?\", \"Explain the first chapter\", or \"What are the key points?\"",
            "key_points": [],
        })

    # Get top relevant passage (main answer)
    answer = get_relevant_context(_pdf_text, req.question, max_chars=800)

    sentences = re.split(r'(?<=[.!?])\s+', answer)
    sentences = [s.strip() for s in sentences if s.strip()]
    key_points = [s for s in sentences if len(s) > 40]

    # Main explanation: full sentences up to ~600 chars (no mid-sentence cut)
    max_main_chars = 600
    main_parts = []
    total = 0
    for s in sentences:
        if total + len(s) + (1 if main_parts else 0) <= max_main_chars:
            main_parts.append(s)
            total += len(s) + (1 if main_parts else 0)
        else:
            break
    main = " ".join(main_parts) if main_parts else answer[:max_main_chars].strip()
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

    audio_bytes = await audio.read()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    print(f"[STT] Received audio: {len(audio_bytes)} bytes, suffix={suffix}")

    try:
        transcript = stt.transcribe(tmp_path)
        if not transcript:
            # Empty = silent recording or too short — tell the user clearly
            return JSONResponse({
                "transcript": "",
                "success": False,
                "detail": "No speech detected. Please speak clearly and try again."
            }, status_code=200)
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
