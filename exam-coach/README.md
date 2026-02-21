# 🌵 ExamGuard — Exam-Mode Privacy Coach

> **100% on-device AI exam practice coach.**  
> Record your answer → get a score + feedback — all locally, no cloud, no login, no data ever leaves your device.

---

## Demo Flow

```
Open app → tap Record → speak answer → tap Stop
     ↓
Whisper (on-device STT) → transcript
     ↓
FunctionGemma (on-device LLM) → Score + Bullet feedback
     ↓
Results shown on screen — nothing sent anywhere
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| AI Inference | Cactus SDK (`libcactus.dylib`) |
| Speech-to-Text | Whisper-tiny via `cactus_transcribe` |
| LLM Feedback | FunctionGemma-270M via `cactus_complete` |
| Backend | Python + FastAPI |
| Frontend | Vanilla HTML/CSS/JS (works offline) |
| Audio | Web Audio API → WAV (no ffmpeg needed) |

---

## Project Structure

```
exam-coach/
├── app.py                  ← FastAPI server  (Person 1 — Vidushi)
├── stt.py                  ← Whisper STT     (Person 1 — Vidushi)
├── coach.py                ← LLM feedback    (Person 1 — Vidushi)
├── requirements.txt        ← Python deps
├── run.sh                  ← One-command startup
├── templates/
│   └── index.html          ← Mobile UI       (Person 2 — Friend)
└── README.md
```

---

## Person 1 — Vidushi (Backend)

### What you own
| File | Responsibility |
|---|---|
| `stt.py` | Loads Whisper-tiny, transcribes WAV audio on-device |
| `coach.py` | Loads FunctionGemma, returns structured score + bullets |
| `app.py` | FastAPI server exposing `/transcribe` and `/feedback` |

### Setup

**Step 1 — Download Whisper model (first time only)**
```bash
cd /Users/vidushi/PycharmProjects/cactus
source ./setup
cactus download openai/whisper-tiny --token YOUR_HF_TOKEN
```

**Step 2 — Install Python dependencies**
```bash
cd exam-coach
pip install -r requirements.txt
```

**Step 3 — Run the app**
```bash
./run.sh
# OR manually:
python app.py
```
App runs at → **http://localhost:8000**

### API Endpoints

#### `POST /transcribe`
Receives a WAV audio file, returns on-device transcript.

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "audio=@recording.wav"
```
```json
{ "transcript": "The main cause of World War One was...", "success": true }
```

#### `POST /feedback`
Receives transcript text, returns score + improvement bullets.

```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"transcript": "The main cause was the assassination..."}'
```
```json
{
  "score": "7/10",
  "bullets": [
    "Add more specific dates and names.",
    "Explain the chain of alliances more clearly.",
    "Conclude with the broader impact."
  ],
  "success": true
}
```

#### `GET /health`
Quick status check.
```json
{ "status": "ok", "stt_ready": true, "llm_ready": true, "on_device": true, "cloud_calls": 0 }
```

### How it works (backend)

```
stt.py
  └── SpeechToText.transcribe(audio_path)
        └── cactus_init(whisper-tiny)
            cactus_transcribe(model, wav_path, prompt)
            cactus_destroy(model)
            → returns transcript string

coach.py
  └── ExamCoach.get_feedback(transcript)
        └── cactus_init(functiongemma-270m-it)
            cactus_complete(model, messages, tools=[give_feedback], force_tools=True)
            cactus_destroy(model)
            → returns { score, bullets }

app.py
  └── FastAPI routes
        POST /transcribe → saves upload → stt.transcribe() → JSON
        POST /feedback   → coach.get_feedback() → JSON
```

### Testing individual components
```bash
# Test STT alone
python stt.py

# Test LLM feedback alone
python coach.py
```

---

## Person 2 — Friend (Frontend)

### What you own
| File | Responsibility |
|---|---|
| `templates/index.html` | Complete mobile-friendly recording UI |

### Setup
```bash
git clone git@github.com:sakotiya/Cactus-hackathon.git
git checkout vid_test
cd Cactus-hackathon/exam-coach
```
No extra installs needed — it's pure HTML/CSS/JS.

To preview the UI (Vidushi must have the server running):
```
http://localhost:8000
```

### UI States to implement / customise

| State | What shows |
|---|---|
| **Idle** | Green "Record" button, instruction text |
| **Recording** | Red pulsing button, live timer (max 60s) |
| **Processing** | Spinner, "Running Whisper on-device" / "Running LLM on-device" |
| **Results** | Transcript card + Score circle + Bullet feedback cards |
| **Error** | Red error message, back to idle |

### How the UI talks to the backend

```javascript
// Step 1 — Send recorded WAV to /transcribe
const formData = new FormData();
formData.append("audio", wavBlob, "recording.wav");
const res  = await fetch("/transcribe", { method: "POST", body: formData });
const { transcript } = await res.json();

// Step 2 — Send transcript to /feedback
const res2 = await fetch("/feedback", {
  method:  "POST",
  headers: { "Content-Type": "application/json" },
  body:    JSON.stringify({ transcript }),
});
const { score, bullets } = await res2.json();
// score  → "7/10"
// bullets → ["suggestion 1", "suggestion 2", "suggestion 3"]
```

### Audio recording (already implemented)
The UI uses the **Web Audio API** to:
1. Capture raw PCM at 16 kHz
2. Encode as WAV entirely in the browser
3. Send WAV to `/transcribe` — no ffmpeg, no external libraries

### Things you can customise
- Color scheme (CSS variables at top of `<style>`)
- App name / logo / tagline in the header
- Privacy badge text
- Card layouts, font sizes, animations
- Add a "question prompt" input field for the user to type the exam question

---

## Running Together (Merge)

```bash
# Vidushi pushes backend changes
git add app.py stt.py coach.py requirements.txt run.sh
git commit -m "backend: <description>"
git push origin vid_test

# Friend pushes frontend changes
git add templates/index.html
git commit -m "ui: <description>"
git push origin vid_test
```

---

## Privacy Guarantee

| What | Status |
|---|---|
| Audio sent to cloud | ❌ Never |
| Transcript sent to cloud | ❌ Never |
| Login / account required | ❌ None |
| External HTTP calls from backend | ❌ Zero |
| All AI runs locally | ✅ Yes |

---

## Hackathon Judging Criteria

This project targets all three qualitative rubrics:

| Rubric | How we address it |
|---|---|
| **Quality of hybrid routing algorithm** | `functiongemma-hackathon/main.py` — query decomposition + smart cloud fallback |
| **End-to-end products with function calls** | ExamGuard uses `cactus_complete` with structured tool calling for feedback |
| **Voice-to-action products with `cactus_transcribe`** | ExamGuard: mic → Whisper → LLM → on-screen results |
