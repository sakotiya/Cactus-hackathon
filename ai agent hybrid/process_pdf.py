#!/usr/bin/env python3
"""
RAG Ingestion Pipeline.

Usage:
  python process_pdf.py                  # ingest all files in ./memory/
  python process_pdf.py path/to/file     # ingest one specific file
  python process_pdf.py path/to/folder   # ingest all files in that folder

Supported formats: .pdf  .txt  .md  .docx

Strategy:
  - Text is extracted from every file
  - Split into overlapping 500-char chunks
  - Each chunk embedded with nomic-embed-text
  - Stored in ChromaDB documents collection (HNSW cosine index)
  FunctionGemma is NOT used here — full text is preserved verbatim.
"""

import os
import sys
import hashlib
import time
from colorama import init, Fore, Style

init(autoreset=True)

SUPPORTED = {".pdf", ".txt", ".md", ".markdown", ".docx"}
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 100
MEMORY_FOLDER = "./memory"


# ------------------------------------------------------------------ #
# Text extraction
# ------------------------------------------------------------------ #

def extract_text(filepath: str) -> str:
    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext == ".pdf":
            from PyPDF2 import PdfReader
            reader = PdfReader(filepath)
            return "\n".join((p.extract_text() or "").strip() for p in reader.pages)

        elif ext == ".docx":
            from docx import Document
            doc = Document(filepath)
            return "\n".join(p.text for p in doc.paragraphs)

        else:  # .txt / .md / .markdown
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
    except Exception as e:
        print(f"{Fore.RED}  Read error: {e}{Style.RESET_ALL}")
        return ""


# ------------------------------------------------------------------ #
# Chunking
# ------------------------------------------------------------------ #

def split_into_chunks(text: str, size: int = CHUNK_SIZE,
                      overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        end   = start + size
        chunk = text[start:end]
        # Snap to last word boundary
        if end < len(text) and not text[end].isspace():
            last_space = chunk.rfind(" ")
            if last_space > size // 2:
                chunk = chunk[:last_space]
        chunk = chunk.strip()
        if chunk:
            chunks.append(chunk)
        start += max(size - overlap, 1)
    return chunks


# ------------------------------------------------------------------ #
# Single-file ingestion
# ------------------------------------------------------------------ #

def ingest(filepath: str, vs=None) -> int:
    """
    Ingest one file into ChromaDB.
    Returns number of chunks stored.
    Skips files already indexed (by content hash).
    """
    from vector_store import VectorStore

    filename = os.path.basename(filepath)
    ext      = os.path.splitext(filename)[1].lower()

    if ext not in SUPPORTED:
        return 0

    if vs is None:
        vs = VectorStore(persist_directory="./data/chroma_db")

    # Check if already ingested (same source name)
    existing = vs.get_document_count(source=filename)
    if existing > 0:
        print(f"  {Fore.YELLOW}Already indexed ({existing} chunks): {filename}{Style.RESET_ALL}")
        return 0

    print(f"\n{Fore.CYAN}📄 Ingesting: {filename}{Style.RESET_ALL}")

    text = extract_text(filepath)
    if not text.strip():
        print(f"  {Fore.RED}No extractable text.{Style.RESET_ALL}")
        return 0

    print(f"  Extracted {len(text):,} chars")

    chunks = split_into_chunks(text)
    total  = len(chunks)
    print(f"  {total} chunks  (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    file_hash = hashlib.md5(text.encode()).hexdigest()[:8]
    success   = 0

    for i, chunk in enumerate(chunks, 1):
        chunk_id = f"{file_hash}_{i:05d}"
        metadata = {"source": filename, "chunk_idx": i, "total": total}

        # Progress bar
        if i == 1 or i % 50 == 0 or i == total:
            pct = i / total * 100
            bar = "█" * int(pct // 5) + "░" * (20 - int(pct // 5))
            print(f"\r  [{bar}] {i}/{total} ({pct:.0f}%)  ", end="", flush=True)

        try:
            vs.add_document_chunk(chunk_id, chunk, metadata)
            success += 1
        except Exception as e:
            print(f"\n  {Fore.RED}Chunk {i} error: {e}{Style.RESET_ALL}")

        if i % 100 == 0:
            time.sleep(0.1)

    print(f"\n  {Fore.GREEN}✅ {success}/{total} chunks indexed{Style.RESET_ALL}")
    return success


# ------------------------------------------------------------------ #
# Folder ingestion
# ------------------------------------------------------------------ #

def ingest_folder(folder: str):
    from vector_store import VectorStore

    folder = os.path.abspath(folder)
    files  = [
        os.path.join(folder, f)
        for f in sorted(os.listdir(folder))
        if os.path.isfile(os.path.join(folder, f))
        and os.path.splitext(f)[1].lower() in SUPPORTED
        and not f.startswith(".")
    ]

    if not files:
        print(f"{Fore.YELLOW}No supported files found in {folder}{Style.RESET_ALL}")
        print(f"Supported: {', '.join(sorted(SUPPORTED))}")
        return

    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"📂  RAG Ingestion — {folder}")
    print(f"    {len(files)} file(s) found")
    print(f"{'='*60}{Style.RESET_ALL}")

    # Share one VectorStore across all files
    vs = VectorStore(persist_directory="./data/chroma_db")

    total_chunks = 0
    for filepath in files:
        total_chunks += ingest(filepath, vs=vs)

    print(f"\n{Fore.GREEN}{'='*60}")
    print(f"✅  All done!  {total_chunks} new chunks embedded")
    print(f"   Total in index: {vs.get_document_count()} chunks")
    print(f"   HNSW params: cosine / M=32 / ef_construction=200")
    print(f"{'='*60}{Style.RESET_ALL}\n")


# ------------------------------------------------------------------ #
# CLI entry point
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else MEMORY_FOLDER

    if os.path.isdir(target):
        ingest_folder(target)
    elif os.path.isfile(target):
        from vector_store import VectorStore
        vs = VectorStore(persist_directory="./data/chroma_db")
        n  = ingest(target, vs=vs)
        print(f"\nTotal in index: {vs.get_document_count()} chunks")
    else:
        print(f"Not found: {target}")
        sys.exit(1)
