"""
Conflict Detector using FunctionGemma (tool calling)
Fast keyword path first, then FunctionGemma for complex cases.
"""

import ollama
from typing import List, Dict, Any


class ConflictDetector:
    """Detect contradictions between facts using FunctionGemma tool calling"""

    def __init__(self, model: str = "functiongemma"):
        self.model = model

    def detect_conflicts(self, new_fact: str, existing_facts: List[Dict[str, Any]],
                         category: str) -> List[str]:
        """
        Detect if new_fact conflicts with existing facts.

        Returns:
            List of conflicting fact IDs
        """
        if not existing_facts:
            return []

        conflicts = self._keyword_based_detection(new_fact, existing_facts, category)
        if conflicts:
            return conflicts

        return self._llm_based_detection(new_fact, existing_facts)

    def _keyword_based_detection(self, new_fact: str,
                                 existing_facts: List[Dict[str, Any]],
                                 category: str) -> List[str]:
        """Fast keyword-based conflict detection (unchanged)"""
        new_lower = new_fact.lower()
        conflicts = []

        patterns = {
            'careers': [
                ('works at', 'work at', 'working at', 'employed at'),
                ('job at', 'position at')
            ],
            'personal': [
                ('lives in', 'live in', 'living in', 'located in'),
                ('name is', 'called', 'i am', "i'm")
            ],
            'interests': [
                ('favorite', 'favourite'),
                ('likes', 'like', 'loves', 'love'),
                ('hates', 'hate', 'dislikes', 'dislike')
            ],
            'food': [
                ('favorite food', 'favorite dish'),
                ('likes to eat', 'loves to eat'),
                ('hates', 'hate', 'dislikes')
            ],
            'sports': [
                ('favorite player', 'favorite team'),
                ('supports', 'support', 'fan of')
            ]
        }

        category_patterns = patterns.get(category, [])

        for fact in existing_facts:
            fact_lower = fact['content'].lower()
            for pattern_group in category_patterns:
                new_has_pattern = any(p in new_lower for p in pattern_group)
                fact_has_pattern = any(p in fact_lower for p in pattern_group)
                if new_has_pattern and fact_has_pattern:
                    conflicts.append(fact['id'])
                    break

        return conflicts

    def _llm_based_detection(self, new_fact: str,
                              existing_facts: List[Dict[str, Any]]) -> List[str]:
        """FunctionGemma tool-calling conflict detection for complex cases"""
        if len(existing_facts) > 10:
            return []

        def report_conflicts(conflict_indexes: list) -> dict:
            """
            Report which existing facts conflict with the new fact.
            A conflict means two facts contradict each other or one replaces the other.

            Args:
                conflict_indexes: List of 0-based indexes of conflicting existing facts.
                                  Use an empty list if there are no conflicts.

            Examples of conflicts:
              - "Works at X" vs "Works at Y"  (different current jobs)
              - "Lives in X" vs "Lives in Y"  (different current locations)

            NOT conflicts:
              - "Worked at X" vs "Works at Y" (past vs current)
              - "Likes X" vs "Likes Y"        (can like multiple things)
            """
            return {}

        existing_text = "\n".join(
            [f"{i}. {fact['content']}" for i, fact in enumerate(existing_facts)]
        )

        prompt = (
            f"Existing facts:\n{existing_text}\n\n"
            f"New fact: \"{new_fact}\"\n\n"
            "Which existing facts (by index) does the new fact conflict with?\n"
            "Call report_conflicts() with the list of conflicting indexes."
        )

        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                tools=[report_conflicts]
            )

            if response.message.tool_calls:
                args = response.message.tool_calls[0].function.arguments
                indexes = args.get("conflict_indexes", [])
                conflict_ids = []
                for idx in indexes:
                    if isinstance(idx, int) and 0 <= idx < len(existing_facts):
                        conflict_ids.append(existing_facts[idx]['id'])
                return conflict_ids

        except Exception as e:
            print(f"LLM conflict detection error: {e}")

        return []
