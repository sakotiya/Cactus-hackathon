"""
Context Engine — assembles the best possible context for each query.

Retrieval order (highest priority first):
  1. Session memory  — facts from the current conversation
  2. Short-term facts — recent personal facts (ChromaDB shortterm, HNSW)
  3. Long-term facts  — durable personal facts (ChromaDB longterm, HNSW)
  4. RAG chunks       — relevant passages from ingested documents
                        (ChromaDB documents, HNSW cosine, M=32)
"""

from typing import List, Dict, Any, Tuple
from memory_manager import MemoryManager
from category_classifier import CategoryClassifier


class ContextEngine:

    def __init__(self, memory_manager: MemoryManager,
                 category_classifier: CategoryClassifier):
        self.memory_manager      = memory_manager
        self.category_classifier = category_classifier

    # ------------------------------------------------------------------ #
    # Main retrieval
    # ------------------------------------------------------------------ #

    def retrieve_context(self, query: str, max_tokens: int = 2000) -> str:
        """
        Build context string from all memory tiers + RAG documents.
        Returns empty string if nothing relevant found.
        """
        # 1. Classify query to guide memory search
        category, _ = self.category_classifier.classify(query)

        # 2. Personal fact retrieval (short + long term)
        short_results = self.memory_manager.search_memories(
            query, category=category, memory_type="short", top_k=5
        )
        long_results = self.memory_manager.search_memories(
            query, category=category, memory_type="long", top_k=5
        )

        # 3. Score, rank, deduplicate personal facts
        all_facts = []
        for m in short_results:
            bonus = 0.1 if m["category"] == category else 0.0
            all_facts.append((m.get("relevance_score", 0.5) + bonus, m))
        for m in long_results:
            bonus = 0.1 if m["category"] == category else 0.0
            all_facts.append((m.get("relevance_score", 0.4) + bonus, m))

        all_facts.sort(key=lambda x: x[0], reverse=True)
        unique_facts = self._deduplicate(all_facts)

        # 4. RAG — semantic search over document chunks (HNSW)
        doc_chunks: List[Dict[str, Any]] = []
        try:
            doc_chunks = self.memory_manager.vector_store.search_documents(
                query, top_k=5
            )
        except Exception:
            pass

        return self._format_context(unique_facts, doc_chunks,
                                    category, max_tokens)

    # ------------------------------------------------------------------ #
    # Deduplication
    # ------------------------------------------------------------------ #

    def _deduplicate(self, scored: List[Tuple[float, Dict]]) -> List[Tuple[float, Dict]]:
        seen, unique = set(), []
        for score, m in scored:
            words = frozenset(m["content"].lower().split())
            duplicate = any(
                len(words & s) / max(len(words | s), 1) > 0.8
                for s in seen
            )
            if not duplicate:
                seen.add(words)
                unique.append((score, m))
        return unique

    # ------------------------------------------------------------------ #
    # Formatting
    # ------------------------------------------------------------------ #

    def _format_context(self,
                        facts:      List[Tuple[float, Dict]],
                        doc_chunks: List[Dict],
                        category:   str,
                        max_tokens: int) -> str:

        parts: List[str] = []

        # --- Personal facts ---
        short = [(s, m) for s, m in facts if m["memory_type"] == "short"]
        long  = [(s, m) for s, m in facts if m["memory_type"] == "long"]

        if short:
            parts.append("=== Personal Memory (recent) ===")
            for _, m in short[:5]:
                tag = f"[{m['category']}] " if m["category"] != category else ""
                parts.append(f"  • {tag}{m['content']}")

        if long and len(short) < 3:
            parts.append("\n=== Personal Memory (long-term) ===")
            for _, m in long[:3]:
                parts.append(f"  • [{m['category']}] {m['content']}")

        # --- RAG document passages ---
        if doc_chunks:
            parts.append("\n=== Relevant Document Passages ===")
            for chunk in doc_chunks[:5]:
                meta   = chunk.get("metadata", {})
                source = meta.get("source", "document")
                idx    = meta.get("chunk_idx", "")
                label  = f"{source} §{idx}" if idx else source
                score  = chunk.get("similarity", 0)
                parts.append(f"[{label}  sim={score:.2f}]\n{chunk['content']}\n")

        if not parts:
            return ""

        context = "\n".join(parts)

        # Trim to token budget (1 token ≈ 4 chars)
        char_limit = max_tokens * 4
        if len(context) > char_limit:
            context = context[:char_limit] + "\n[...context truncated]"

        return context

    # ------------------------------------------------------------------ #
    # Category view (for /category command)
    # ------------------------------------------------------------------ #

    def get_category_context(self, category: str, limit: int = 10) -> str:
        memories = self.memory_manager.get_memories_by_category(category)
        if not memories:
            return f"No memories in category: {category}"

        memories.sort(key=lambda m: m.get("timestamp", ""), reverse=True)
        parts = [f"=== {category.upper()} ===\n"]

        short = [m for m in memories if m["memory_type"] == "short"]
        long  = [m for m in memories if m["memory_type"] == "long"]

        if short:
            parts.append("Recent:")
            for m in short[:limit]:
                parts.append(f"  • {m['content']}")
        if long:
            parts.append("\nLong-term:")
            for m in long[:limit]:
                parts.append(f"  • {m['content']}")

        return "\n".join(parts)
