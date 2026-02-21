"""
stt.py — On-device Speech-to-Text using Cactus Whisper
=======================================================
PERSON 1 (YOU) owns this file.

Uses the Cactus FFI to run Whisper locally.
No audio leaves the device.
"""

import sys
import json
import os
from pathlib import Path

# Cactus Python bindings
CACTUS_REPO = Path(__file__).parent.parent.parent / "cactus"
sys.path.insert(0, str(CACTUS_REPO / "python" / "src"))

from cactus import cactus_init, cactus_transcribe, cactus_destroy

# Whisper model weights path
WHISPER_PATH = str(CACTUS_REPO / "weights" / "whisper-tiny")

# Whisper prompt for English transcription
WHISPER_PROMPT = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"


class SpeechToText:
    """On-device Whisper transcription via Cactus."""

    def __init__(self, model_path: str = WHISPER_PATH):
        self.model_path = model_path
        self._check_model()

    def _check_model(self):
        if not Path(self.model_path).exists():
            raise RuntimeError(
                f"Whisper model not found at: {self.model_path}\n"
                f"Run:  cactus download openai/whisper-tiny\n"
                f"From: {CACTUS_REPO}"
            )

    def transcribe(self, audio_path: str) -> str:
        """
        Transcribe a WAV audio file on-device.

        Args:
            audio_path: Path to a WAV file (16kHz mono recommended)

        Returns:
            Transcribed text string
        """
        model = cactus_init(self.model_path)
        try:
            raw = cactus_transcribe(model, audio_path, prompt=WHISPER_PROMPT)
            result = json.loads(raw)
            if not result.get("success"):
                raise RuntimeError(f"Transcription failed: {result.get('error')}")
            return (result.get("response") or "").strip()
        finally:
            cactus_destroy(model)


# ── Quick smoke-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile, wave, struct, math

    # Generate a 1-second silent WAV for testing model load
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(tmp.name, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 16000)

    print(f"Testing Whisper model at: {WHISPER_PATH}")
    stt = SpeechToText()
    text = stt.transcribe(tmp.name)
    print(f"Transcript: '{text}' (empty for silence — model works!)")
    os.unlink(tmp.name)
