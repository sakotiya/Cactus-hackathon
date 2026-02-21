#!/bin/bash
# ─────────────────────────────────────────────────────────
#  ExamGuard — run script
#  Usage: ./run.sh
# ─────────────────────────────────────────────────────────

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CACTUS_DIR="$(cd "$SCRIPT_DIR/../../cactus" && pwd)"

echo ""
echo "  ╔══════════════════════════════════════╗"
echo "  ║   🌵 ExamGuard — Privacy Coach       ║"
echo "  ╚══════════════════════════════════════╝"
echo ""

# 1. Activate cactus venv
source "$CACTUS_DIR/setup" 2>/dev/null || true

# 2. Install python deps
pip install -q -r "$SCRIPT_DIR/requirements.txt"

# 3. Check whisper model
WHISPER="$CACTUS_DIR/weights/whisper-tiny"
if [ ! -d "$WHISPER" ]; then
  echo "  Downloading Whisper model (first run only)…"
  cd "$CACTUS_DIR"
  cactus download openai/whisper-tiny
fi

# 4. Start server
cd "$SCRIPT_DIR"
echo "  Starting server → http://localhost:8000"
echo ""
python app.py
