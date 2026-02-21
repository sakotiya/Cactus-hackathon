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


class ExamCoach:
    """On-device exam answer evaluator via Cactus FunctionGemma."""

    def __init__(self, model_path: str = LLM_PATH):
        self.model_path = model_path

    def get_feedback(self, transcript: str) -> dict:
        """
        Evaluate a student's spoken answer and return structured feedback.

        Args:
            transcript: The student's spoken answer as text

        Returns:
            {
                "score":   "7/10",
                "bullets": ["suggestion 1", "suggestion 2", "suggestion 3"],
                "raw":     <raw model output for debugging>
            }
        """
        if not transcript.strip():
            return {
                "score": "N/A",
                "bullets": ["No speech detected. Please try recording again."],
                "raw": "",
            }

        prompt = (
            f"Student's answer:\n\"{transcript}\"\n\n"
            f"Score this answer from 1-10 on clarity, structure, and completeness. "
            f"Give 2-3 short improvement suggestions."
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

        return self._parse(raw_str)

    def _parse(self, raw_str: str) -> dict:
        """Parse Cactus function-call output into a clean feedback dict."""
        try:
            raw = json.loads(raw_str)
        except json.JSONDecodeError:
            return self._fallback(raw_str)

        calls = raw.get("function_calls", [])
        if calls:
            args = calls[0].get("arguments", {})
            score_val = args.get("score", "?")
            bullets = [
                args[k]
                for k in ("suggestion_1", "suggestion_2", "suggestion_3")
                if args.get(k)
            ]
            return {
                "score":   f"{score_val}/10",
                "bullets": bullets or ["Keep practising!"],
                "raw":     raw_str,
            }

        # Fallback: try to parse free-form response text
        return self._fallback(raw.get("response") or raw_str)

    def _fallback(self, text: str) -> dict:
        """Extract score and bullets from free-form text if structured parse fails."""
        score = "?"
        m = re.search(r"(\d+)\s*/\s*10", str(text))
        if m:
            score = f"{m.group(1)}/10"
        bullets = re.findall(r"[-•*]\s+(.+)", str(text))[:3]
        if not bullets:
            bullets = ["Try to be more structured.", "Add specific examples.", "Be more concise."]
        return {"score": score, "bullets": bullets, "raw": str(text)}


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
