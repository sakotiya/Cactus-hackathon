"""
Gourav LLM Agent

Local (offline):
  - functiongemma — all chat + all tool calls (agentic RAG loop)
  - nomic-embed-text + ChromaDB HNSW — embeddings & retrieval

Cloud (online, only when live data is needed):
  - Gemini via Portkey + Google Search  (triggered by search_web tool)

Memory tiers:
  - Session    — in-memory facts from the current conversation
  - Short-term — recent personal facts (ChromaDB shortterm, HNSW)
  - Long-term  — durable facts (ChromaDB longterm, HNSW)
  - Documents  — RAG chunks from files + chat history (ChromaDB documents, HNSW)
"""

import google.genai as genai
from google.genai import types as genai_types
import ollama
import threading
import queue
import hashlib
import datetime
from typing import Optional, List, Dict, Any
from memory_manager import MemoryManager
from category_classifier import CategoryClassifier
from fact_extractor import FactExtractor
from conflict_detector import ConflictDetector
from context_engine import ContextEngine
from file_watcher import FileWatcher
from session_manager import SessionManager


class HybridAgent:

    def __init__(self,
                 gemini_model: str = "gemini-2.5-flash",
                 function_model: str = "functiongemma",
                 system_prompt: Optional[str] = None,
                 data_dir: str = "./data"):
        """
        Args:
            gemini_model:   Cloud model — used ONLY when search_web tool is called
            function_model: Local model — all chat + all tool calls
            data_dir:       Data directory for ChromaDB + memories.json
        """
        self.gemini_model   = gemini_model
        self.function_model = function_model
        self.chat_model     = function_model   # alias so _local_chat still works
        self.system_prompt  = system_prompt or self._default_system_prompt()

        # Gemini client — direct API, used for final answer generation
        self._gemini_client = genai.Client(api_key="AIzaSyAVp80ixdWFr28iam_iqTMA_vwXDZ2HtdE")

        # Memory stack
        self.memory_manager      = MemoryManager(data_dir=data_dir,
                                                  gemma_model=function_model)
        self.category_classifier = CategoryClassifier(model=function_model)
        self.fact_extractor      = FactExtractor(model=function_model)
        self.conflict_detector   = ConflictDetector(model=function_model)
        self.context_engine      = ContextEngine(self.memory_manager,
                                                  self.category_classifier)

        # Session memory (in-memory, current conversation)
        self.session_manager = SessionManager()
        self.session_manager.start_new_session()

        # File watcher for memory/ folder
        self.file_watcher = FileWatcher(
            memory_folder="./memory",
            process_callback=self._process_new_file
        )

        # Stats
        self.local_responses  = 0
        self.online_responses = 0

        # Async chat-memory writer
        # Queue items: (chunk_id, text, metadata)
        self._mem_queue: queue.Queue = queue.Queue()
        self._mem_worker = threading.Thread(
            target=self._memory_writer_loop, daemon=True
        )
        self._mem_worker.start()

        self._ensure_models()

    # ------------------------------------------------------------------ #
    # System prompt
    # ------------------------------------------------------------------ #

    def _default_system_prompt(self) -> str:
        return (
            "You are Gourav LLM, an intelligent assistant with long-term memory.\n\n"
            "You have access to:\n"
            "  - Personal memories (careers, sports, interests, pop culture, food, tech)\n"
            "  - Document knowledge from ingested PDFs (retrieved via RAG)\n\n"
            "Use the context provided below to answer accurately.\n"
            "Be concise and conversational."
        )

    # ------------------------------------------------------------------ #
    # Startup
    # ------------------------------------------------------------------ #

    def _ensure_models(self):
        """Ensure functiongemma and nomic-embed-text are available."""
        for model in (self.function_model, "nomic-embed-text"):
            try:
                ollama.show(model)
            except Exception:
                print(f"Pulling {model} ...")
                ollama.pull(model)

    # ------------------------------------------------------------------ #
    # Async chat-memory writer
    # ------------------------------------------------------------------ #

    def _memory_writer_loop(self):
        """
        Background daemon thread — drains _mem_queue and writes chunks
        to ChromaDB without blocking the chat loop.
        """
        vs = self.memory_manager.vector_store
        while True:
            try:
                item = self._mem_queue.get(timeout=1)
                if item is None:          # shutdown signal
                    break
                chunk_id, text, metadata = item
                vs.add_document_chunk(chunk_id, text, metadata)
                self._mem_queue.task_done()
            except queue.Empty:
                continue
            except Exception:
                pass                      # never crash the background thread

    def _store_chat_async(self, user_message: str, response_text: str):
        """
        Enqueue one conversation turn for async embedding into ChromaDB.
        Returns immediately — embedding happens in the background.
        """
        ts       = datetime.datetime.now().isoformat(timespec="seconds")
        combined = f"User: {user_message}\nAssistant: {response_text}"
        chunk_id = "chat_" + hashlib.md5(combined.encode()).hexdigest()[:12]
        metadata = {
            "source":    "chat_history",
            "timestamp": ts,
            "type":      "conversation",
        }
        self._mem_queue.put((chunk_id, combined, metadata))

    def start_file_watcher(self):
        self.file_watcher.start_background(scan_existing=True)

    # ------------------------------------------------------------------ #
    # File ingestion callback (called by file watcher)
    # ------------------------------------------------------------------ #

    def _process_new_file(self, filepath: str, content: str, file_hash: str):
        """
        Called by the file watcher for every new/changed file in memory/.

        All supported files (.pdf, .txt, .md, .docx) are ingested as RAG
        chunks into ChromaDB (HNSW index) so their full text is searchable.

        Non-PDF text files also run FunctionGemma fact extraction to pull
        out personal facts into the short/long-term memory tiers.
        """
        import os
        filename = os.path.basename(filepath)
        ext      = os.path.splitext(filename)[1].lower()
        supported = {".pdf", ".txt", ".md", ".markdown", ".docx"}

        if ext not in supported:
            return

        # --- RAG ingestion (all supported types) ---
        from process_pdf import ingest
        vs = self.memory_manager.vector_store
        ingest(filepath, vs=vs)

        # --- Personal fact extraction (text/docx only, not PDF) ---
        # PDFs like yearbooks are reference material, not personal documents.
        if ext != ".pdf" and content:
            facts = self.fact_extractor.extract_from_document(content, filename)
            if facts:
                print(f"   📝 {len(facts)} personal facts from {filename}")
                for fact_data in facts:
                    self.memory_manager.add_memory(
                        content=fact_data["content"],
                        category=fact_data["category"],
                        memory_type="long",
                        metadata={
                            "source":     filename,
                            "importance": fact_data["importance"],
                            "file_hash":  file_hash,
                        }
                    )

    # ------------------------------------------------------------------ #
    # Agentic tool definitions
    # ------------------------------------------------------------------ #

    def _tool_search_documents(self, query: str) -> str:
        """Search ingested FILES only (PDFs, txt, docx) — excludes chat history."""
        results = self.memory_manager.vector_store.search_documents(
            query, top_k=6, exclude_chat_history=True
        )
        if not results:
            return "No relevant document passages found."
        parts = []
        for r in results:
            src   = r["metadata"].get("source", "doc")
            idx   = r["metadata"].get("chunk_idx", "")
            label = f"{src} §{idx}" if idx else src
            parts.append(f"[{label}  sim={r['similarity']:.2f}]\n{r['content']}")
        return "\n\n".join(parts)

    def _tool_search_chat_history(self, query: str) -> str:
        """Search past conversations only."""
        results = self.memory_manager.vector_store.search_chat_history(query, top_k=3)
        if not results:
            return ""
        parts = []
        for r in results:
            parts.append(f"[past conversation  sim={r['similarity']:.2f}]\n{r['content']}")
        return "\n\n".join(parts)

    def _tool_search_memory(self, query: str) -> str:
        """Search personal short-term and long-term memory facts."""
        short = self.memory_manager.search_memories(query, memory_type="short", top_k=4)
        long  = self.memory_manager.search_memories(query, memory_type="long",  top_k=3)
        session = self.session_manager.get_temporal_context()
        parts = []
        if session:
            parts.append(f"[Session]\n{session}")
        for m in short:
            parts.append(f"[short-term/{m['category']}] {m['content']}")
        for m in long:
            parts.append(f"[long-term/{m['category']}] {m['content']}")
        return "\n".join(parts) if parts else "No personal memory found."

    def _tool_search_web(self, query: str) -> str:
        """Search the web for live / real-time information via Gemini+Google."""
        try:
            params: Dict[str, Any] = {
                "model":       self.gemini_model,
                "messages":    [{"role": "user", "content": query}],
                "temperature": 0.3,
                "max_tokens":  512,
                "tools": [{"type": "function", "function": {
                    "name": "googleSearch", "parameters": {"timeRangeFilter": None}
                }}]
            }
            resp = self.portkey.chat.completions.create(**params)
            self.online_responses += 1
            return resp.choices[0].message.content or "No web result."
        except Exception as e:
            return f"Web search error: {e}"

    # ------------------------------------------------------------------ #
    # FunctionGemma tools — local retrieval ONLY (no web)
    # ------------------------------------------------------------------ #

    def _get_retrieval_tools(self) -> List[Dict]:
        """Tools given to FunctionGemma: search local DB only."""
        return [
            {
                "type": "function",
                "function": {
                    "name":        "search_documents",
                    "description": (
                        "Search ingested files (PDFs, yearbooks, notes) using "
                        "semantic similarity. Use for any question about facts, "
                        "events, awards, people, or content from uploaded files."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string",
                                      "description": "Specific search query"}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name":        "search_memory",
                    "description": (
                        "Search personal memory (short-term, long-term, session). "
                        "Use for questions about the user's personal info, "
                        "preferences, history, or past conversations."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string",
                                      "description": "What to look up in memory"}
                        },
                        "required": ["query"]
                    }
                }
            }
        ]

    def _execute_retrieval_tool(self, name: str, args: Dict) -> str:
        """Execute a local retrieval tool and return results as text."""
        query = args.get("query", "")
        if name == "search_documents":
            print(f"\n   📚 search_documents({query!r})", flush=True)
            return self._tool_search_documents(query)
        elif name == "search_memory":
            print(f"\n   🧠 search_memory({query!r})", flush=True)
            return self._tool_search_memory(query)
        return f"Unknown tool: {name}"

    # ------------------------------------------------------------------ #
    # Phase 1 — deterministic context retrieval (no LLM deciding)
    # ------------------------------------------------------------------ #

    def _gather_context(self, user_message: str,
                        debug: bool = False) -> str:
        """
        Always search both local collections directly — no LLM in the loop.
        FunctionGemma (270M) is too small to reliably decide when to call tools,
        so we do it unconditionally and let Gemini filter what's relevant.
        """
        parts = []

        # 1. Document store (PDFs, files) — always search
        q_preview = user_message[:50]

        # 1. Ingested files (PDFs, docs) — highest priority, chat history excluded
        print(f"\n   📚 search_documents({q_preview!r})", flush=True)
        doc_results = self._tool_search_documents(user_message)
        if doc_results and "No relevant" not in doc_results:
            parts.append(f"=== Document passages ===\n{doc_results}")

        # 2. Personal memory (short + long term + session)
        print(f"   🧠 search_memory({q_preview!r})", flush=True)
        mem_results = self._tool_search_memory(user_message)
        if mem_results and "No personal memory" not in mem_results:
            parts.append(f"=== Personal memory ===\n{mem_results}")

        # 3. Past conversations — lowest priority
        chat_results = self._tool_search_chat_history(user_message)
        if chat_results:
            parts.append(f"=== Past conversations ===\n{chat_results}")

        context = "\n\n".join(parts)

        if debug and context:
            print(f"\n--- Context ({len(context)} chars) ---\n{context[:600]}...\n")

        return context

    # ------------------------------------------------------------------ #
    # Phase 2 — Gemini generates the final answer from retrieved context
    # ------------------------------------------------------------------ #

    def _answer_with_gemini(self, user_message: str, context: str,
                            stream: bool = False) -> str:
        """
        Answer using retrieved local context only — no web search.
        Gemini is instructed to stay strictly within the provided context.
        """
        if context:
            system = (
                f"{self.system_prompt}\n\n"
                f"Answer the user's question using ONLY the context below. "
                f"Do not use any external knowledge. "
                f"If the answer is not in the context, say so clearly.\n\n"
                f"=== Retrieved Context ===\n{context}"
            )
        else:
            system = (
                f"{self.system_prompt}\n\n"
                f"No relevant context was found in the local knowledge base. "
                f"Answer from your general knowledge if you can, or say you don't know."
            )
        try:
            if stream:
                text = ""
                for chunk in self._gemini_client.models.generate_content_stream(
                    model=self.gemini_model,
                    contents=user_message,
                    config=genai_types.GenerateContentConfig(
                        system_instruction=system,
                        temperature=0.3,
                        max_output_tokens=1024,
                    )
                ):
                    token = chunk.text or ""
                    print(token, end="", flush=True)
                    text += token
                print()
                return text
            else:
                resp = self._gemini_client.models.generate_content(
                    model=self.gemini_model,
                    contents=user_message,
                    config=genai_types.GenerateContentConfig(
                        system_instruction=system,
                        temperature=0.3,
                        max_output_tokens=1024,
                    )
                )
                return resp.text
        except Exception as e:
            return f"[Gemini error: {e}]"

    def _answer_with_gemini_web(self, user_message: str,
                                stream: bool = False) -> str:
        """
        Gemini with Google Search grounding — for live / real-time queries only.
        Triggered by /web command or keywords like 'latest', 'live score', etc.
        """
        try:
            if stream:
                text = ""
                for chunk in self._gemini_client.models.generate_content_stream(
                    model=self.gemini_model,
                    contents=user_message,
                    config=genai_types.GenerateContentConfig(
                        system_instruction=self.system_prompt,
                        temperature=0.7,
                        max_output_tokens=1024,
                        tools=[genai_types.Tool(
                            google_search=genai_types.GoogleSearch()
                        )],
                    )
                ):
                    token = chunk.text or ""
                    print(token, end="", flush=True)
                    text += token
                print()
                return text
            else:
                resp = self._gemini_client.models.generate_content(
                    model=self.gemini_model,
                    contents=user_message,
                    config=genai_types.GenerateContentConfig(
                        system_instruction=self.system_prompt,
                        temperature=0.7,
                        max_output_tokens=1024,
                        tools=[genai_types.Tool(
                            google_search=genai_types.GoogleSearch()
                        )],
                    )
                )
                return resp.text
        except Exception as e:
            return f"[Gemini Search error: {e}]"

    # ------------------------------------------------------------------ #
    # Main chat
    # ------------------------------------------------------------------ #

    def _needs_online_search(self, query: str) -> bool:
        """True only for explicitly live/real-time queries."""
        keywords = [
            "latest", "right now", "live score", "live match",
            "weather", "stock price", "breaking news", "today's news",
            "current price", "real-time"
        ]
        return any(kw in query.lower() for kw in keywords)

    def chat(self, user_message: str,
             auto_extract_facts: bool = False,
             stream: bool = False,
             debug: bool = False) -> str:

        self.session_manager.add_message_to_session("user", user_message)

        if self._needs_online_search(user_message):
            # Explicit live query — skip local DB, go straight to Gemini+Search
            print(f"🌐 [Gemini + Search] ", end="", flush=True)
            response_text = self._answer_with_gemini_web(user_message, stream=stream)
            self.online_responses += 1
        else:
            # Phase 1 — search local DB directly (always)
            print(f"⚡ [FunctionGemma] retrieving context...", flush=True)
            context = self._gather_context(user_message, debug=debug)

            # Phase 2 — Gemini answers strictly from retrieved context
            print(f"\n🟡 [Gemini] ", end="", flush=True)
            response_text = self._answer_with_gemini(user_message, context, stream=stream)
            self.local_responses += 1

        self.session_manager.add_message_to_session("assistant", response_text)
        self._store_chat_async(user_message, response_text)

        if auto_extract_facts:
            self._extract_and_store_session(user_message, response_text)

        return response_text

    # ------------------------------------------------------------------ #
    # llama3.1 — direct chat (used internally, no tools)
    # ------------------------------------------------------------------ #

    def _local_chat(self, messages: List[Dict[str, str]],
                    stream: bool = False) -> str:
        try:
            if stream:
                text = ""
                for chunk in ollama.chat(model=self.chat_model,
                                         messages=messages, stream=True):
                    token = chunk["message"]["content"]
                    print(token, end="", flush=True)
                    text += token
                print()
                return text
            else:
                resp = ollama.chat(model=self.chat_model, messages=messages)
                return resp["message"]["content"]
        except Exception as e:
            return f"[{self.chat_model} error: {e}]"

    # ------------------------------------------------------------------ #
    # Gemini — online search fallback
    # ------------------------------------------------------------------ #

    def _gemini_search(self, messages: List[Dict[str, str]],
                       stream: bool = False) -> str:
        try:
            params: Dict[str, Any] = {
                "model":       self.gemini_model,
                "messages":    messages,
                "temperature": 0.7,
                "max_tokens":  1024,
                "tools": [{
                    "type": "function",
                    "function": {
                        "name":       "googleSearch",
                        "parameters": {"timeRangeFilter": None}
                    }
                }]
            }
            if stream:
                text = ""
                params["stream"] = True
                for chunk in self.portkey.chat.completions.create(**params):
                    if chunk.choices[0].delta.content:
                        token = chunk.choices[0].delta.content
                        print(token, end="", flush=True)
                        text += token
                print()
                return text
            else:
                resp = self.portkey.chat.completions.create(**params)
                return resp.choices[0].message.content
        except Exception as e:
            return f"[Gemini error: {e}]"

    # ------------------------------------------------------------------ #
    # Fact extraction into session memory
    # ------------------------------------------------------------------ #

    def _extract_and_store_session(self, user_msg: str, agent_resp: str):
        print(f"\n⚡ [FunctionGemma] Extracting facts...")
        facts = self.fact_extractor.extract_facts(user_msg, agent_resp)
        if facts:
            print(f"🔍 {len(facts)} fact(s) → session memory")
            for f in facts:
                self.session_manager.add_temporal_fact(
                    f"[{f['category']}] {f['content']}"
                )
                print(f"   💭 [{f['category']}] {f['content'][:60]}...")
            print(f"✅ Commit with /end\n")
        else:
            print(f"ℹ️  No new facts.\n")

    # ------------------------------------------------------------------ #
    # Session commit (session → short-term → long-term via decay)
    # ------------------------------------------------------------------ #

    def end_session_and_commit(self, silent: bool = False) -> str:
        session_facts = self.session_manager.get_session_facts()

        if not session_facts:
            self.session_manager.end_current_session()
            return "✅ Session ended (no facts to commit)"

        if not silent:
            print(f"\n🔄 Committing {len(session_facts)} session facts...")

        committed = 0
        for fact_str in session_facts:
            if "]" in fact_str:
                category = fact_str.split("[")[1].split("]")[0]
                content  = fact_str.split("] ", 1)[1]
            else:
                category, content = "general", fact_str

            existing  = self.memory_manager.get_memories_by_category(
                category, memory_type="short"
            )
            conflicts = self.conflict_detector.detect_conflicts(
                content, existing, category
            )
            for cid in conflicts:
                cm = self.memory_manager.get_memory(cid)
                if cm and not silent:
                    print(f"   ⚠️  Conflict archived: {cm['content'][:40]}...")
                self.memory_manager.archive_memory(cid)

            self.memory_manager.add_memory(
                content=content, category=category, memory_type="short",
                metadata={"importance": 0.7, "from_session": True}
            )
            committed += 1
            if not silent:
                print(f"   ✅ [{category}] {content[:50]}...")

        self.session_manager.end_current_session()
        if not silent:
            print(f"\n✅ {committed} facts committed to short-term memory!\n")
        return f"✅ Session ended, {committed} facts committed"

    # ------------------------------------------------------------------ #
    # Memory management helpers
    # ------------------------------------------------------------------ #

    def remember(self, fact: str, category: Optional[str] = None,
                 is_current: bool = True) -> str:
        if not category:
            category, _ = self.category_classifier.classify(fact)
        self.memory_manager.add_memory(
            content=fact, category=category,
            memory_type="short" if is_current else "long"
        )
        return f"✅ Remembered [{category}]: {fact}"

    def forget(self, pattern: str) -> str:
        archived = self.memory_manager.forget_pattern(pattern)
        if archived:
            lines = "\n".join(f"  - {x[:60]}" for x in archived[:5])
            extra = f"\n  ... and {len(archived)-5} more" if len(archived) > 5 else ""
            return f"🧹 Forgot {len(archived)} item(s):\n{lines}{extra}"
        return f"⚠️  No memories matching '{pattern}'"

    def get_category_memories(self, category: str) -> str:
        return self.context_engine.get_category_context(category)

    def run_decay(self) -> str:
        stats = self.memory_manager.run_decay_process()
        return (f"🔄 Decay complete: {stats['promoted']} promoted, "
                f"{stats['archived']} archived")

    def get_memory_stats(self) -> Dict[str, Any]:
        stats = self.memory_manager.get_statistics()
        stats["local_responses"]  = self.local_responses
        stats["online_responses"] = self.online_responses
        vs = self.memory_manager.vector_store
        stats["document_chunks"]  = vs.get_document_count()
        stats["chat_chunks"]      = vs.get_document_count(source="chat_history")
        stats["pending_writes"]   = self._mem_queue.qsize()
        return stats

    def change_model(self, new_model: str) -> str:
        old = self.gemini_model
        self.gemini_model = new_model
        return f"✅ Gemini model: '{old}' → '{new_model}'"
