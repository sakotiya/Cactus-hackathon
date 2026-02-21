"""
Vector Store — ChromaDB backed by HNSW index.

Three collections:
  shortterm  — recent personal facts (high churn)
  longterm   — durable personal facts
  documents  — raw document chunks for RAG (HNSW-tuned for recall)

All embeddings use nomic-embed-text (via Ollama).
"""

import uuid
import chromadb
from chromadb.config import Settings
import ollama
from typing import List, Dict, Any, Optional
import os


class VectorStore:

    # HNSW knobs — tune recall vs. speed
    _HNSW_MEMORY = {
        "hnsw:space":           "cosine",
        "hnsw:M":               16,    # neighbours per node; 16 is a good default
        "hnsw:construction_ef": 100,   # build quality
        "hnsw:search_ef":       50,    # query recall
    }
    _HNSW_DOCUMENTS = {
        "hnsw:space":           "cosine",
        "hnsw:M":               32,    # denser graph for large corpora
        "hnsw:construction_ef": 200,
        "hnsw:search_ef":       100,
    }

    def __init__(self, persist_directory: str = "./data/chroma_db",
                 gemma_model: str = "functiongemma"):
        os.makedirs(persist_directory, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        # Personal memory collections
        self.shortterm_collection = self.client.get_or_create_collection(
            name="shortterm", metadata=self._HNSW_MEMORY
        )
        self.longterm_collection = self.client.get_or_create_collection(
            name="longterm", metadata=self._HNSW_MEMORY
        )
        # Document RAG collection (denser HNSW for large corpora)
        self.documents_collection = self.client.get_or_create_collection(
            name="documents", metadata=self._HNSW_DOCUMENTS
        )

        self.embedding_model = "nomic-embed-text"
        self.gemma_model = gemma_model
        self._ensure_models()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _ensure_models(self):
        try:
            ollama.show(self.embedding_model)
        except Exception:
            print(f"Pulling embedding model: {self.embedding_model}")
            ollama.pull(self.embedding_model)

    def _embed(self, text: str) -> List[float]:
        try:
            return ollama.embeddings(model=self.embedding_model, prompt=text)["embedding"]
        except Exception as e:
            print(f"Embedding error: {e}")
            return [0.0] * 768  # nomic-embed-text dim

    def _format_results(self, results: Dict) -> List[Dict[str, Any]]:
        formatted = []
        if not results["ids"] or not results["ids"][0]:
            return formatted
        for i in range(len(results["ids"][0])):
            formatted.append({
                "id":         results["ids"][0][i],
                "content":    results["documents"][0][i],
                "metadata":   results["metadatas"][0][i],
                "similarity": 1 - results["distances"][0][i],
            })
        return formatted

    # ------------------------------------------------------------------ #
    # Short-term / long-term memory (personal facts)
    # ------------------------------------------------------------------ #

    def add_to_shortterm(self, memory_id: str, content: str,
                         metadata: Dict[str, Any]) -> None:
        self.shortterm_collection.add(
            ids=[memory_id],
            embeddings=[self._embed(content)],
            documents=[content],
            metadatas=[metadata]
        )

    def add_to_longterm(self, memory_id: str, content: str,
                        metadata: Dict[str, Any]) -> None:
        self.longterm_collection.add(
            ids=[memory_id],
            embeddings=[self._embed(content)],
            documents=[content],
            metadatas=[metadata]
        )

    def search_shortterm(self, query: str, category: Optional[str] = None,
                         top_k: int = 5) -> List[Dict[str, Any]]:
        if self.shortterm_collection.count() == 0:
            return []
        results = self.shortterm_collection.query(
            query_embeddings=[self._embed(query)],
            n_results=min(top_k, self.shortterm_collection.count()),
            where={"category": category} if category else None
        )
        return self._format_results(results)

    def search_longterm(self, query: str, category: Optional[str] = None,
                        top_k: int = 5) -> List[Dict[str, Any]]:
        if self.longterm_collection.count() == 0:
            return []
        results = self.longterm_collection.query(
            query_embeddings=[self._embed(query)],
            n_results=min(top_k, self.longterm_collection.count()),
            where={"category": category} if category else None
        )
        return self._format_results(results)

    def delete_from_shortterm(self, memory_id: str) -> None:
        try:
            self.shortterm_collection.delete(ids=[memory_id])
        except Exception:
            pass

    def delete_from_longterm(self, memory_id: str) -> None:
        try:
            self.longterm_collection.delete(ids=[memory_id])
        except Exception:
            pass

    def move_to_longterm(self, memory_id: str, content: str,
                         metadata: Dict[str, Any]) -> None:
        self.add_to_longterm(memory_id, content, metadata)
        self.delete_from_shortterm(memory_id)

    # ------------------------------------------------------------------ #
    # Document RAG (HNSW-tuned)
    # ------------------------------------------------------------------ #

    def add_document_chunk(self, chunk_id: str, text: str,
                           metadata: Dict[str, Any]) -> None:
        """Embed and store one document chunk. Upsert-safe."""
        embedding = self._embed(text)
        existing = self.documents_collection.get(ids=[chunk_id])
        if existing["ids"]:
            self.documents_collection.update(
                ids=[chunk_id], embeddings=[embedding],
                documents=[text], metadatas=[metadata]
            )
        else:
            self.documents_collection.add(
                ids=[chunk_id], embeddings=[embedding],
                documents=[text], metadatas=[metadata]
            )

    def search_documents(self, query: str, top_k: int = 6,
                         source: Optional[str] = None,
                         exclude_chat_history: bool = False) -> List[Dict[str, Any]]:
        """
        Hybrid search over document chunks:
          1. Semantic HNSW search  — finds conceptually relevant chunks
          2. Keyword search        — finds chunks containing exact words from the query
                                     (crucial for proper nouns, names, codes)
        Results are merged and deduplicated, ranked by best score.
        """
        n = self.documents_collection.count()
        if n == 0:
            return []

        if source:
            where = {"source": source}
        elif exclude_chat_history:
            where = {"source": {"$ne": "chat_history"}}
        else:
            where = None

        seen_ids: set = set()
        merged:   List[Dict[str, Any]] = []

        # --- 1. Semantic search (HNSW) ---
        try:
            sem_results = self.documents_collection.query(
                query_embeddings=[self._embed(query)],
                n_results=min(top_k, n),
                where=where
            )
            for item in self._format_results(sem_results):
                if item["id"] not in seen_ids:
                    seen_ids.add(item["id"])
                    merged.append(item)
        except Exception:
            pass

        # --- 2. Keyword search — find chunks by exact text match ---
        # Stop words to skip — too generic, produce noise
        STOP = {
            "what", "which", "where", "when", "does", "have", "with", "that",
            "this", "about", "from", "tell", "give", "list", "find", "show",
            "many", "much", "some", "more", "most", "also", "well", "just",
            "your", "their", "there", "they", "them", "then", "than", "been"
        }
        # Collect meaningful words (≥4 chars, not stop words), ordered longest first
        meaningful = sorted(
            {w.strip("?.,!") for w in query.split()
             if len(w.strip("?.,!")) >= 4 and w.lower().strip("?.,!") not in STOP},
            key=len, reverse=True
        )

        # Build (term, score) pairs — longer terms = higher score (more specific)
        term_score: List[tuple] = []
        for w in meaningful:
            base_score = min(0.95, 0.75 + len(w) * 0.01)  # "Kolkata"=0.82, "Knight"=0.81
            for variant in {w, w.lower(), w.capitalize(), w.title()}:
                term_score.append((variant, base_score))
        # Also try exact full query
        term_score.append((query, 0.98))

        for term, score in term_score:
            try:
                kw_results = self.documents_collection.get(
                    where_document={"$contains": term},
                    limit=top_k * 3
                )
                if not kw_results["ids"]:
                    continue
                for i, doc_id in enumerate(kw_results["ids"]):
                    meta = kw_results["metadatas"][i]
                    if exclude_chat_history and meta.get("source") == "chat_history":
                        continue
                    if source and meta.get("source") != source:
                        continue
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        merged.append({
                            "id":         doc_id,
                            "content":    kw_results["documents"][i],
                            "metadata":   meta,
                            "similarity": score,
                        })
                    else:
                        # Upgrade score if this hit ranks higher
                        for item in merged:
                            if item["id"] == doc_id and item["similarity"] < score:
                                item["similarity"] = score
                                break
            except Exception:
                pass

        # Sort by similarity score, return top_k
        merged.sort(key=lambda x: x["similarity"], reverse=True)
        return merged[:top_k]

    def search_chat_history(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Semantic search over past conversations only."""
        n = self.documents_collection.count()
        if n == 0:
            return []
        results = self.documents_collection.query(
            query_embeddings=[self._embed(query)],
            n_results=min(top_k, n),
            where={"source": "chat_history"}
        )
        return self._format_results(results)

    def get_document_count(self, source: Optional[str] = None) -> int:
        if source:
            return len(self.documents_collection.get(where={"source": source})["ids"])
        return self.documents_collection.count()

    # ------------------------------------------------------------------ #
    # Counts
    # ------------------------------------------------------------------ #

    def get_count(self, collection: str = "shortterm") -> int:
        if collection == "shortterm":
            return self.shortterm_collection.count()
        elif collection == "longterm":
            return self.longterm_collection.count()
        elif collection == "documents":
            return self.documents_collection.count()
        return 0
