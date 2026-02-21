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

# ── Cactus path setup ────────────────────────────────────────────────────────
CACTUS_REPO = Path(__file__).parent.parent.parent / "cactus"
sys.path.insert(0, str(CACTUS_REPO / "python" / "src"))
from cactus import cactus_init, cactus_rag_query, cactus_destroy, cactus_reset

from stt import SpeechToText
from coach import ExamCoach
from pdf_reader import extract_text

LLM_PATH = str(CACTUS_REPO / "weights" / "functiongemma-270m-it")

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

# ── RAG state ────────────────────────────────────────────────────────────────
# When a PDF is uploaded we write its text to a temp directory and init a
# cactus model with corpus_dir pointing there. cactus handles all chunking,
# embedding, and retrieval — no manual prompting.
_rag_model   = None          # cactus model handle with RAG corpus
_rag_dir     = None          # temp dir path (corpus.txt lives here)
_pdf_text    = ""            # raw text (for fallback / quiz)
_pdf_name    = ""

TEMPLATE_PATH = Path(__file__).parent / "templates" / "index.html"
print("\nExamGuard is running at http://localhost:8000\n")


def _init_rag(corpus_dir: str):
    """(Re-)initialize the RAG model pointing at corpus_dir."""
    global _rag_model
    if _rag_model is not None:
        try:
            cactus_destroy(_rag_model)
        except Exception:
            pass
    _rag_model = cactus_init(LLM_PATH, corpus_dir, False)


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    if not TEMPLATE_PATH.exists():
        return HTMLResponse("<h1>UI not found</h1>")
    return HTMLResponse(content=TEMPLATE_PATH.read_text())


@app.get("/health")
async def health():
    return {
        "status":    "ok",
        "stt_ready": stt is not None,
        "rag_ready": _rag_model is not None,
        "pdf_name":  _pdf_name or None,
        "on_device": True,
    }


@app.post("/upload-pdf")
async def upload_pdf(pdf: UploadFile = File(...)):
    """
    Extract PDF text, write to corpus.txt, re-init cactus RAG model.
    After this, /ask queries use cactus_rag_query — no prompting.
    """
    global _rag_dir, _pdf_text, _pdf_name

    if not pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Save PDF to temp file for extraction
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(await pdf.read())
        tmp_pdf = tmp.name

    try:
        text = extract_text(tmp_pdf)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.unlink(tmp_pdf)
        except OSError:
            pass

    # Write corpus to a persistent temp directory
    if _rag_dir and Path(_rag_dir).exists():
        shutil.rmtree(_rag_dir, ignore_errors=True)
    _rag_dir = tempfile.mkdtemp(prefix="examguard_rag_")
    corpus_file = Path(_rag_dir) / "corpus.txt"
    corpus_file.write_text(text, encoding="utf-8")

    # Re-init cactus with the new corpus (builds embedding index automatically)
    try:
        _init_rag(_rag_dir)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG init failed: {e}")

    _pdf_text = text
    _pdf_name = pdf.filename

    return JSONResponse({"success": True, "filename": pdf.filename, "chars": len(text)})


class AskRequest(BaseModel):
    question: str


@app.post("/ask")
async def ask(req: AskRequest):
    """
    Study mode: use cactus built-in RAG to retrieve relevant chunks.
    Returns the top-k chunks as the answer — no prompting needed.
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Please ask a question.")

    if _rag_model is None:
        # No PDF loaded — return a helpful message
        return JSONResponse({
            "success":    True,
            "answer":     "Please upload your study PDF first using the button at the top.",
            "key_points": [],
            "chunks":     [],
        })

    try:
        # cactus_rag_query returns [{"score": float, "text": str}, ...]
        chunks = cactus_rag_query(_rag_model, req.question, top_k=4)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not chunks:
        return JSONResponse({
            "success":    True,
            "answer":     "No relevant content found in your PDF for this question. Try rephrasing.",
            "key_points": [],
            "chunks":     [],
        })

    # Build answer: combine top chunks, sorted by relevance score
    chunks_sorted = sorted(chunks, key=lambda c: c.get("score", 0), reverse=True)
    texts = [c["text"].strip() for c in chunks_sorted if c.get("text", "").strip()]

    # Main answer = highest-scoring chunk
    answer = texts[0] if texts else ""
    # Key points = subsequent chunks (already relevant, just different sections)
    key_points = texts[1:4]

    return JSONResponse({
        "success":    True,
        "answer":     answer,
        "key_points": key_points,
        "chunks":     len(chunks),
    })


class QuizRequest(BaseModel):
    topic: str = ""


@app.post("/quiz")
async def quiz(req: QuizRequest):
    """
    Practice mode: generate a question using RAG to find a relevant passage,
    then pick an exam-worthy sentence from it.
    """
    import re

    if _rag_model is not None and _pdf_text:
        query = req.topic if req.topic.strip() else "main concept definition"
        try:
            chunks = cactus_rag_query(_rag_model, query, top_k=2)
        except Exception:
            chunks = []

        for chunk in chunks:
            text = chunk.get("text", "").strip()
            sentences = re.split(r'(?<=[.!?])\s+', text)
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
    if _rag_model is not None and (req.question or req.transcript):
        try:
            query  = req.question or req.transcript
            chunks = cactus_rag_query(_rag_model, query, top_k=2)
            pdf_context = " ".join(c.get("text", "") for c in chunks)[:600]
        except Exception:
            pass

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
