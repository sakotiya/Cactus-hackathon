"""
coach.py — On-device Exam Feedback using Cactus + FunctionGemma
================================================================
PERSON 1 (YOU) owns this file.

Takes a spoken answer transcript and returns a score + bullet-point
suggestions — all running locally via Cactus, no cloud required.
"""

import sys
import json
import re
from pathlib import Path

# Cactus Python bindings
CACTUS_REPO = Path(__file__).parent.parent.parent / "cactus"
sys.path.insert(0, str(CACTUS_REPO / "python" / "src"))

from cactus import cactus_init, cactus_complete, cactus_destroy

# FunctionGemma model weights path (already downloaded)
LLM_PATH = str(CACTUS_REPO / "weights" / "functiongemma-270m-it")

# ── Exam feedback tool definition ────────────────────────────────────────────────
# We use FunctionGemma's function-calling strength to get structured output
# reliably from a 270M model — much more reliable than free-form generation.

FEEDBACK_TOOL = {
    "name": "give_feedback",
    "description": "Give structured feedback on a student's spoken exam answer",
    "parameters": {
        "type": "object",
        "properties": {
            "score": {
                "type": "integer",
                "description": "Score from 1 to 10 based on clarity, structure, and completeness",
            },
            "suggestion_1": {
                "type": "string",
                "description": "First improvement suggestion (one short sentence)",
            },
            "suggestion_2": {
                "type": "string",
                "description": "Second improvement suggestion (one short sentence)",
            },
            "suggestion_3": {
                "type": "string",
                "description": "Third improvement suggestion (one short sentence)",
            },
        },
        "required": ["score", "suggestion_1", "suggestion_2"],
    },
}

SYSTEM_PROMPT = (
    "You are an exam coach. "
    "Evaluate the student's answer and call give_feedback with a score (1-10) "
    "and 2-3 short improvement suggestions."
)

# Fallback suggestions when the model gives nonsense output
_DEFAULT_SUGGESTIONS = [
    "Structure your answer with a clear introduction, body, and conclusion.",
    "Use specific examples or evidence to support your points.",
    "Be more concise — aim for clarity over length.",
]


class ExamCoach:
    """On-device exam answer evaluator via Cactus FunctionGemma."""

    def __init__(self, model_path: str = LLM_PATH):
        self.model_path = model_path

    def get_feedback(self, transcript: str,
                     question: str = "",
                     pdf_context: str = "") -> dict:
        """
        Evaluate a student's spoken answer and return structured feedback.

        Args:
            transcript:  The student's spoken answer as text
            question:    The exam question they were answering (optional)
            pdf_context: Relevant excerpt from their study PDF (optional)

        Returns:
            {
                "score":   "7/10",
                "bullets": ["suggestion 1", "suggestion 2", "suggestion 3"],
                "raw":     <raw model output for debugging>
            }
        """
        transcript = transcript.strip()
        if not transcript:
            return {
                "score": "N/A",
                "bullets": ["No answer detected. Please try recording or typing again."],
                "raw": "",
            }

        # Detect if the "answer" is just a question / too short to evaluate
        word_count = len(transcript.split())
        if word_count < 5:
            score_val = max(1, min(3, word_count))
            return {
                "score": f"{score_val}/10",
                "bullets": [
                    "Your answer is too short — aim for at least 2-3 full sentences.",
                    "Explain your reasoning, not just the conclusion.",
                    "Add examples or definitions to support your answer.",
                ],
                "raw": "",
            }

        # Keep the prompt SHORT so the small 270M model doesn't get confused.
        # PDF context is compressed to a brief hint (≤200 chars) rather than
        # full paragraphs — FunctionGemma degrades badly with long context.
        context_hint = ""
        if pdf_context:
            # Take just the first 200 chars as a topic hint
            hint = pdf_context[:200].replace("\n", " ").strip()
            context_hint = f" The topic covers: {hint}..."
        if question:
            context_hint += f" The question asked: {question}"

        prompt = (
            f"Student answer: \"{transcript}\""
            f"{context_hint}\n\n"
            "Call give_feedback with score (1-10) and 2-3 improvement tips."
        )

        model = cactus_init(self.model_path)
        try:
            raw_str = cactus_complete(
                model,
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                tools=[{"type": "function", "function": FEEDBACK_TOOL}],
                force_tools=True,
                max_tokens=300,
                stop_sequences=["<|im_end|>", "<end_of_turn>"],
            )
        finally:
            cactus_destroy(model)

        return self._parse(raw_str, transcript, question)

    def _parse(self, raw_str: str, transcript: str = "", question: str = "") -> dict:
        """Parse Cactus function-call output into a clean feedback dict."""
        try:
            raw = json.loads(raw_str)
        except json.JSONDecodeError:
            return self._fallback(raw_str, transcript, question)

        calls = raw.get("function_calls", [])
        if calls:
            args = calls[0].get("arguments", {})
            score_val = args.get("score", "?")

            # Validate score is a real number
            try:
                score_int = int(score_val)
                if not (1 <= score_int <= 10):
                    raise ValueError
            except (ValueError, TypeError):
                score_int = None

            bullets = [
                args[k]
                for k in ("suggestion_1", "suggestion_2", "suggestion_3")
                if args.get(k)
            ]

            # Sanitize: discard bullets that just echo the transcript or question
            bullets = self._sanitize_bullets(bullets, transcript, question)

            if score_int and bullets:
                return {
                    "score":   f"{score_int}/10",
                    "bullets": bullets,
                    "raw":     raw_str,
                }

        # Fallback: try to parse free-form response text
        return self._fallback(raw.get("response") or raw_str, transcript, question)

    def _sanitize_bullets(self, bullets: list, transcript: str, question: str) -> list:
        """Remove suggestions that just repeat the user's input."""
        clean = []
        bad_refs = {transcript.lower().strip(), question.lower().strip()} - {""}
        for b in bullets:
            b_stripped = b.strip()
            b_lower = b_stripped.lower().rstrip("?.")
            # Skip if bullet is essentially the same as the transcript/question
            if any(b_lower in ref or ref in b_lower for ref in bad_refs):
                continue
            # Skip very short or empty bullets
            if len(b_stripped) < 10:
                continue
            clean.append(b_stripped)
        return clean or _DEFAULT_SUGGESTIONS

    def _fallback(self, text: str, transcript: str = "", question: str = "") -> dict:
        """Extract score and bullets from free-form text if structured parse fails."""
        score = "?"
        m = re.search(r"(\d+)\s*/\s*10", str(text))
        if m:
            val = int(m.group(1))
            if 1 <= val <= 10:
                score = f"{val}/10"

        bullets = re.findall(r"[-•*]\s+(.+)", str(text))[:3]
        bullets = self._sanitize_bullets(bullets, transcript, question)
        return {"score": score, "bullets": bullets, "raw": str(text)}


    # ── Study mode: explain a topic from PDF ──────────────────────────────────
    def explain(self, question: str, pdf_context: str = "") -> dict:
        """
        Answer a study question using PDF context.

        Strategy: FunctionGemma (270M) is a function-calling model, not a
        text-generation model — asking it to write explanations causes refusals.
        Instead we:
          1. Return the most relevant PDF excerpt directly as the explanation
             (accurate, no hallucination, always works).
          2. Use FunctionGemma only to extract 2-3 key points from that text
             (a simple copy/extraction task it handles well).
        """
        if not question.strip():
            return {"answer": "Please ask a question.", "key_points": []}

        if not pdf_context:
            return {
                "answer": (
                    "No PDF loaded yet. Upload your study PDF using the button at the top, "
                    "then ask your question — the explanation will come directly from your material."
                ),
                "key_points": ["Upload a PDF to get started."],
            }

        # ── Step 1: use PDF text directly as the explanation ──────────────────
        # Clean up whitespace and take the most relevant passage (already ranked
        # by pdf_reader.get_relevant_context before this is called).
        clean = re.sub(r'\s+', ' ', pdf_context).strip()
        explanation = clean[:900]   # show up to 900 chars from the PDF

        # ── Step 2: ask FunctionGemma to extract key points from that text ───
        # This is a copy/extraction task — much easier than free-form generation.
        KEY_TOOL = {
            "name": "extract_points",
            "description": "Extract the most important facts from the given text",
            "parameters": {
                "type": "object",
                "properties": {
                    "point_1": {"type": "string", "description": "First important fact (one sentence)"},
                    "point_2": {"type": "string", "description": "Second important fact (one sentence)"},
                    "point_3": {"type": "string", "description": "Third important fact (one sentence, optional)"},
                },
                "required": ["point_1", "point_2"],
            },
        }

        prompt = (
            f"Text:\n{clean[:500]}\n\n"
            f"Extract the 2-3 most important facts from this text. "
            f"Call extract_points with short sentences."
        )

        key_points = []
        model = cactus_init(self.model_path)
        try:
            raw_str = cactus_complete(
                model,
                [
                    {"role": "system", "content": "You extract key facts from text. Call extract_points."},
                    {"role": "user",   "content": prompt},
                ],
                tools=[{"type": "function", "function": KEY_TOOL}],
                force_tools=True,
                max_tokens=250,
                stop_sequences=["<|im_end|>", "<end_of_turn>"],
            )
            raw = json.loads(raw_str)
            calls = raw.get("function_calls", [])
            if calls:
                args = calls[0].get("arguments", {})
                for k in ("point_1", "point_2", "point_3"):
                    v = (args.get(k) or "").strip()
                    if v and len(v) > 15:
                        key_points.append(v)
        except Exception:
            pass
        finally:
            cactus_destroy(model)

        # Fallback key points: split PDF into sentences and pick first 3
        if not key_points:
            sentences = re.split(r'(?<=[.!?])\s+', clean)
            key_points = [s.strip() for s in sentences if len(s.strip()) > 30][:3]

        if not key_points:
            key_points = ["Review this section in your PDF for more detail."]

        return {"answer": explanation, "key_points": key_points}

    # ── Quiz mode: generate a practice question from PDF ─────────────────────
    def generate_question(self, pdf_context: str, topic: str = "") -> dict:
        """
        Generate a practice exam question from the PDF content.
        Returns { "question": "...", "hint": "..." }
        """
        context = pdf_context[:600] if pdf_context else ""
        topic_hint = f" Focus on: {topic}." if topic else ""

        prompt = (
            f"Study material:\n{context}\n\n"
            f"Generate one exam-style practice question a student can answer verbally.{topic_hint} "
            f"Also give a one-sentence hint."
        )

        QUIZ_TOOL = {
            "name": "make_question",
            "description": "Generate a practice exam question",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "A clear exam-style question",
                    },
                    "hint": {
                        "type": "string",
                        "description": "A one-sentence hint to help the student",
                    },
                },
                "required": ["question", "hint"],
            },
        }

        model = cactus_init(self.model_path)
        try:
            raw_str = cactus_complete(
                model,
                [
                    {"role": "system", "content": "You are an exam coach. Generate practice questions and call make_question."},
                    {"role": "user",   "content": prompt},
                ],
                tools=[{"type": "function", "function": QUIZ_TOOL}],
                force_tools=True,
                max_tokens=200,
                stop_sequences=["<|im_end|>", "<end_of_turn>"],
            )
        finally:
            cactus_destroy(model)

        try:
            raw = json.loads(raw_str)
            calls = raw.get("function_calls", [])
            if calls:
                args = calls[0].get("arguments", {})
                q = args.get("question", "").strip()
                h = args.get("hint", "").strip()
                if q and len(q) > 10 and "?" in q:
                    return {"question": q, "hint": h or "Think about what you read in the PDF."}
        except Exception:
            pass

        # Fallback: build a question from a key sentence in the PDF
        if pdf_context:
            sentences = re.split(r'(?<=[.!?])\s+', re.sub(r'\s+', ' ', pdf_context))
            long_sentences = [s.strip() for s in sentences if len(s.strip()) > 40]
            if long_sentences:
                pick = long_sentences[0]
                return {
                    "question": f"Explain the following concept in your own words: \"{pick[:120]}…\"",
                    "hint": "Use specific terms and examples from the text.",
                }

        return {
            "question": "Explain the main concept covered in your study material.",
            "hint": "Use specific terms and examples from the text.",
        }


# ── Quick smoke-test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    coach = ExamCoach()
    sample = (
        "I think the main cause of World War One was the assassination of "
        "Archduke Franz Ferdinand which triggered a chain of alliances and "
        "led to a global conflict."
    )
    print("Testing ExamCoach on sample answer...")
    result = coach.get_feedback(sample)
    print(f"\nScore:   {result['score']}")
    for i, b in enumerate(result["bullets"], 1):
        print(f"  {i}. {b}")
