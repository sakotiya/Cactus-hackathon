"""
app.py — FastAPI Server for ExamGuard Privacy Coach
====================================================
PERSON 1 (YOU) owns this file.

Endpoints:
  GET  /              → serves the mobile UI (index.html)
  POST /transcribe    → audio file → transcript (on-device Whisper)
  POST /feedback      → transcript text → score + bullets (on-device LLM)
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
        "status": "ok",
        "stt_ready":   stt is not None,
        "llm_ready":   True,
        "on_device":   True,
        "cloud_calls": 0,
    }


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


class FeedbackRequest(BaseModel):
    transcript: str


@app.post("/feedback")
async def feedback(req: FeedbackRequest):
    """
    Receive a transcript and return on-device AI feedback.

    Returns: { "score": "7/10", "bullets": ["...", "...", "..."] }
    """
    try:
        result = coach.get_feedback(req.transcript)
        return JSONResponse({
            "score":   result["score"],
            "bullets": result["bullets"],
            "success": True,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Entry point ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
