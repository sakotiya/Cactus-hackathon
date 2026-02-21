# Gourav LLM Agent

A local-first AI agent with long-term memory, hybrid RAG retrieval, and cloud fallback for live queries.

---

## Architecture

```
Your question
     │
     ▼
⚡ FunctionGemma (local)
     ├─ search_documents  →  ChromaDB HNSW  (PDFs, files)
     └─ search_memory     →  ChromaDB HNSW  (personal facts)
           │
           ▼  retrieved context
🟡 Gemini (cloud)  →  generates final answer from context only
           │
           └─ /web or live queries  →  Gemini + Google Search
```

**Memory tiers (all stored in ChromaDB):**

| Tier | Contents | Collection |
|---|---|---|
| Session | Current conversation facts (in-memory until `/end`) | — |
| Short-term | Recent personal facts | `shortterm` |
| Long-term | Durable personal facts | `longterm` |
| Documents | Ingested file chunks + past chat turns | `documents` |

**Search strategy — hybrid (semantic + keyword):**
- HNSW cosine similarity for conceptual matches
- Exact keyword matching for proper nouns, names, codes
- Longer/more specific keywords ranked higher

---

## Requirements

- Python 3.11+
- Ollama 0.16.3+
- Portkey API key (for Gemini access)

---

## Installation

### 1. Install Ollama

```bash
brew install ollama
```

> If already installed, upgrade: `sudo chown -R $USER /opt/homebrew && brew upgrade ollama`

### 2. Start Ollama and pull required models

```bash
ollama serve &
ollama pull functiongemma
ollama pull nomic-embed-text
```

### 3. Clone / navigate to the project

```bash
cd "/path/to/ai agent hybrid"
```

### 4. Create virtual environment and install dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

### Ingest documents (run once per new file)

Drop any `.pdf`, `.txt`, `.md`, or `.docx` file into the `memory/` folder, then:

```bash
source venv/bin/activate
python process_pdf.py          # ingest all files in memory/
python process_pdf.py path/to/file.pdf   # ingest a specific file
```

### Run the agent

```bash
source venv/bin/activate
python main.py
```

---

## Commands

| Command | Description |
|---|---|
| *(just type)* | Ask anything — FunctionGemma retrieves context, Gemini answers |
| `/web <question>` | Force online search via Gemini + Google |
| `/ingest` | Re-scan `memory/` folder and index new files |
| `/ingest <path>` | Ingest a specific file |
| `/remember <fact>` | Manually add a personal fact |
| `/forget <pattern>` | Remove memories matching a pattern |
| `/categories` | Show all memory categories |
| `/category <name>` | View memories in a category |
| `/stats` | Memory, document chunk, and usage stats |
| `/decay` | Run memory decay (promotes old short-term to long-term) |
| `/end` | End session and commit facts to memory |
| `/new` | Start a fresh session |
| `/model <name>` | Switch the Gemini cloud model |
| `DEBUG: <question>` | Show retrieved context before answering |
| `/help` | Show all commands |
| `/quit` | Exit |

---

## Adding knowledge

Just drop files into `memory/` — the file watcher picks them up automatically on startup, or run `/ingest` while the agent is running.

```
memory/
├── Adda247-Yearbook-2025.pdf   ← indexed as RAG chunks
├── my-notes.md                 ← indexed + personal facts extracted
└── resume.docx                 ← indexed + personal facts extracted
```

---

## Models used

| Model | Role | Runs |
|---|---|---|
| `functiongemma` (270M) | Context retrieval — decides what to search | Local |
| `nomic-embed-text` (137M) | Embeddings for HNSW index | Local |
| `Gemini` via Portkey | Final answer generation | Cloud |
| `Gemini + Search` | Live queries (news, scores, weather) | Cloud |
