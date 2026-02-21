"""
app.py — FastAPI Server for ExamGuard Privacy Coach
====================================================
PERSON 1 (YOU) owns this file.

Endpoints:
  GET  /              → serves the mobile UI (index.html)
  POST /transcribe    → audio file → transcript (on-device Whisper)
  POST /feedback      → transcript text → score + bullets (on-device LLM)
  POST /upload-pdf    → PDF file → extracts text, stores in session
  GET  /health        → quick status check

Run with:
  python app.py
Then open: http://localhost:8000
"""

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from stt import SpeechToText
from coach import ExamCoach
from pdf_reader import extract_text, get_relevant_context

# ── App setup ───────────────────────────────────────────────────────────────────
app = FastAPI(title="ExamGuard — Privacy Coach")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models once at startup (not per-request)
print("Loading on-device models...")
try:
    stt   = SpeechToText()
    print("  ✓ Whisper (STT) ready")
except RuntimeError as e:
    stt = None
    print(f"  ✗ Whisper not available: {e}")

coach = ExamCoach()
print("  ✓ FunctionGemma (LLM) ready")
print("\nExamGuard is running at http://localhost:8000\n")

# In-memory PDF store (single session, no DB needed)
_pdf_text: str = ""
_pdf_name: str = ""

# ── HTML template ────────────────────────────────────────────────────────────────
TEMPLATE_PATH = Path(__file__).parent / "templates" / "index.html"


# ── Routes ───────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the mobile UI."""
    if not TEMPLATE_PATH.exists():
        return HTMLResponse("<h1>UI not found — add templates/index.html</h1>")
    return HTMLResponse(content=TEMPLATE_PATH.read_text())


@app.get("/health")
async def health():
    return {
        "status":      "ok",
        "stt_ready":   stt is not None,
        "llm_ready":   True,
        "on_device":   True,
        "cloud_calls": 0,
        "pdf_loaded":  bool(_pdf_text),
        "pdf_name":    _pdf_name or None,
    }


@app.post("/upload-pdf")
async def upload_pdf(pdf: UploadFile = File(...)):
    """
    Upload a study PDF. Text is extracted on-device (no cloud).
    Extracted text is stored in memory for the session.

    Returns: { "success": true, "pages": N, "chars": N, "filename": "..." }
    """
    global _pdf_text, _pdf_name

    if not pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        data = await pdf.read()
        tmp.write(data)
        tmp_path = tmp.name

    try:
        text = extract_text(tmp_path)
        _pdf_text = text
        _pdf_name = pdf.filename
        return JSONResponse({
            "success":  True,
            "filename": pdf.filename,
            "chars":    len(text),
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """
    Receive an audio file (WAV) and return the on-device transcript.

    The browser sends raw WAV bytes encoded by the Web Audio API —
    no ffmpeg needed, no cloud upload.
    """
    if stt is None:
        raise HTTPException(
            status_code=503,
            detail="Whisper model not loaded. Run: cactus download openai/whisper-tiny"
        )

    # Save upload to a temp WAV file
    suffix = ".wav"
    content_type = audio.content_type or ""
    if "mp4" in content_type:
        suffix = ".mp4"
    elif "webm" in content_type:
        suffix = ".webm"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        data = await audio.read()
        tmp.write(data)
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


class AskRequest(BaseModel):
    question: str

class QuizRequest(BaseModel):
    topic: str = ""

class FeedbackRequest(BaseModel):
    transcript: str
    question:   str = ""   # optional: the exam question the student answered


@app.post("/ask")
async def ask(req: AskRequest):
    """
    Study mode: ask a question, get an AI explanation using the uploaded PDF.
    Returns: { "answer": "...", "key_points": [...] }
    """
    try:
        pdf_context = ""
        if _pdf_text:
            pdf_context = get_relevant_context(_pdf_text, req.question, max_chars=800)
        result = coach.explain(question=req.question, pdf_context=pdf_context)
        return JSONResponse({"answer": result["answer"], "key_points": result["key_points"], "success": True})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/quiz")
async def quiz(req: QuizRequest):
    """
    Practice mode: generate a practice question from the uploaded PDF.
    Returns: { "question": "...", "hint": "..." }
    """
    try:
        pdf_context = get_relevant_context(_pdf_text, req.topic, max_chars=600) if _pdf_text else ""
        result = coach.generate_question(pdf_context=pdf_context, topic=req.topic)
        return JSONResponse({"question": result["question"], "hint": result["hint"], "success": True})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback")
async def feedback(req: FeedbackRequest):
    """
    Receive a transcript (and optionally a question) and return on-device AI feedback.
    If a PDF is loaded, finds the most relevant context and includes it in the prompt.

    Returns: { "score": "7/10", "bullets": ["...", "...", "..."] }
    """
    try:
        # Pull relevant PDF excerpt (if a PDF has been uploaded)
        pdf_context = ""
        if _pdf_text:
            search_query = req.question or req.transcript
            pdf_context = get_relevant_context(_pdf_text, search_query, max_chars=1500)

        result = coach.get_feedback(
            transcript=req.transcript,
            question=req.question,
            pdf_context=pdf_context,
        )
        return JSONResponse({
            "score":       result["score"],
            "bullets":     result["bullets"],
            "used_pdf":    bool(pdf_context),
            "success":     True,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Entry point ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
