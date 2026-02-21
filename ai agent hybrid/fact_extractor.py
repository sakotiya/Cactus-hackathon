"""
Fact Extractor using FunctionGemma (tool calling)
Each fact is captured via a tool call, allowing multiple facts per conversation.
"""

import ollama
from typing import List, Dict, Any


class FactExtractor:
    """Extract structured facts from conversations using FunctionGemma tool calling"""

    def __init__(self, model: str = "functiongemma"):
        self.model = model

    def extract_facts(self, user_message: str, agent_response: str) -> List[Dict[str, Any]]:
        """
        Extract facts from a conversation exchange using tool calling.

        Returns:
            List of facts: [{"content", "category", "is_current", "importance"}]
        """
        def store_fact(content: str, category: str, is_current: bool, importance: float) -> dict:
            """
            Store a single extracted fact from the conversation.

            Args:
                content: The fact text
                category: One of: careers, sports, interests, pop_culture,
                          personal, food, tech, general
                is_current: True if this is current/active info (e.g. "I work at X"),
                            False if past (e.g. "I worked at X")
                importance: Importance score 0.0-1.0
                            (0.9-1.0: critical, 0.6-0.8: important, 0.3-0.5: casual)
            """
            return {}

        prompt = (
            "Extract factual information about the user from this conversation.\n"
            "Call store_fact() once for each distinct fact. Extract 0-5 facts.\n"
            "Skip greetings, questions without answers, and vague statements.\n\n"
            f"User: \"{user_message}\"\n"
            f"Assistant: \"{agent_response}\"\n\n"
            "Extract facts now:"
        )

        return self._run_extraction(prompt, store_fact)

    def extract_from_document(self, content: str, source: str) -> List[Dict[str, Any]]:
        """
        Extract facts from a document (resume, notes, etc.) using tool calling.

        Args:
            content: Document content
            source: Source filename

        Returns:
            List of facts
        """
        if len(content) > 4000:
            content = content[:4000] + "\n\n[Content truncated...]"

        def store_fact(content: str, category: str, is_current: bool, importance: float) -> dict:
            """
            Store a single key fact extracted from the document.

            Args:
                content: The fact text
                category: One of: careers, sports, interests, pop_culture,
                          personal, food, tech, general
                is_current: True for current info (present job = True, past job = False)
                importance: Importance score 0.0-1.0
            """
            return {}

        prompt = (
            f"Extract key facts from this document (source: {source}).\n"
            "Call store_fact() once for each fact. Extract 5-15 most important facts.\n"
            "Focus on: personal info, career history, skills, preferences, achievements.\n\n"
            f"Document:\n{content}\n\n"
            "Extract facts now:"
        )

        return self._run_extraction(prompt, store_fact)

    def _run_extraction(self, prompt: str, tool_fn) -> List[Dict[str, Any]]:
        """Run FunctionGemma tool calling and collect all store_fact calls"""
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                tools=[tool_fn]
            )

            facts = []
            if response.message.tool_calls:
                for tool_call in response.message.tool_calls:
                    args = tool_call.function.arguments
                    content_val = args.get("content", "")
                    if not content_val:
                        continue
                    facts.append({
                        "content": content_val,
                        "category": args.get("category", "general"),
                        "is_current": bool(args.get("is_current", True)),
                        "importance": float(args.get("importance", 0.5))
                    })

            return facts

        except Exception as e:
            print(f"Fact extraction error: {e}")
            return []
