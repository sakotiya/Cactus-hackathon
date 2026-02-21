"""
Category Classifier using FunctionGemma (tool calling)
Falls back to keyword matching if tool call fails.
"""

import ollama
from typing import Tuple
from categories import CATEGORIES, keyword_match_category


class CategoryClassifier:
    """Classify text into categories using FunctionGemma tool calling"""

    def __init__(self, model: str = "functiongemma"):
        self.model = model
        self._cache = {}

    def classify(self, text: str, use_cache: bool = True) -> Tuple[str, float]:
        """
        Classify text into a category.

        Returns:
            (category, confidence_score)
        """
        if use_cache and text in self._cache:
            return self._cache[text]

        try:
            category, confidence = self._llm_classify(text)
        except Exception:
            category = keyword_match_category(text)
            confidence = 0.5

        if use_cache:
            self._cache[text] = (category, confidence)

        return category, confidence

    def _llm_classify(self, text: str) -> Tuple[str, float]:
        """Use FunctionGemma tool calling to classify text"""
        categories_list = list(CATEGORIES.keys())

        def set_category(category: str, confidence: float) -> dict:
            """
            Set the category and confidence for the given text.

            Args:
                category: One of: careers, sports, interests, pop_culture,
                          personal, food, tech, general
                confidence: Confidence score between 0.0 and 1.0
            """
            return {"category": category, "confidence": confidence}

        prompt = (
            f"Classify this text into the most appropriate category.\n"
            f"Available categories: careers, sports, interests, pop_culture, personal, food, tech, general\n\n"
            f"- careers: job, work, career information\n"
            f"- sports: sports, athletes, games\n"
            f"- interests: hobbies, preferences, likes/dislikes\n"
            f"- pop_culture: movies, music, entertainment\n"
            f"- personal: name, location, family\n"
            f"- food: food preferences, restaurants\n"
            f"- tech: technology, programming, software\n"
            f"- general: if no specific category fits\n\n"
            f"Text: \"{text}\""
        )

        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            tools=[set_category]
        )

        if response.message.tool_calls:
            args = response.message.tool_calls[0].function.arguments
            category = args.get("category", "general")
            confidence = float(args.get("confidence", 0.7))

            if category not in categories_list:
                category = "general"

            return category, confidence

        raise Exception("FunctionGemma made no tool call; falling back to keyword match")

    def clear_cache(self):
        """Clear classification cache"""
        self._cache = {}
