"""
ingest.py
Loads documents from ./documents, splits them into overlapping chunks,
and stores them (with auto-generated embeddings) in a persistent Chroma collection.

Run standalone to (re)build the index:
    python ingest.py
"""

import os
import glob
import uuid
import chromadb
from pypdf import PdfReader

DOCS_DIR = "documents"
DB_DIR = "chroma_db"
COLLECTION_NAME = "docs"

CHUNK_SIZE = 800       # target characters per chunk
CHUNK_OVERLAP = 150    # overlap between consecutive chunks


def load_text_from_file(path: str) -> str:
    """Extract raw text from .txt, .md, or .pdf files."""
    ext = os.path.splitext(path)[1].lower()

    if ext in (".txt", ".md"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    if ext == ".pdf":
        reader = PdfReader(path)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)

    raise ValueError(f"Unsupported file type: {ext}")


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Simple sliding-window chunker that tries to break on paragraph/sentence
    boundaries where possible, instead of cutting mid-word every time.
    """
    text = " ".join(text.split())
    if not text:
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        if end < text_len:
            window = text[start:end]
            last_period = window.rfind(". ")
            if last_period != -1 and last_period > chunk_size * 0.5:
                end = start + last_period + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap
        if start < 0 or end == text_len:
            break

    return chunks


def build_index():
    """Read every supported file in DOCS_DIR, chunk it, and add to Chroma."""
    client = chromadb.PersistentClient(path=DB_DIR)

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    supported_extensions = ("*.txt", "*.md", "*.pdf")
    files = []
    for pattern in supported_extensions:
        files.extend(glob.glob(os.path.join(DOCS_DIR, pattern)))

    if not files:
        print(f"No documents found in ./{DOCS_DIR}/. Add some .txt, .md, or .pdf files and re-run.")
        return

    all_chunks, all_ids, all_metadatas = [], [], []

    for filepath in files:
        filename = os.path.basename(filepath)
        print(f"Processing {filename}...")

        text = load_text_from_file(filepath)
        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(str(uuid.uuid4()))
            all_metadatas.append({"source": filename, "chunk_index": i})

        print(f"  -> {len(chunks)} chunks")

    if all_chunks:
        collection.add(documents=all_chunks, ids=all_ids, metadatas=all_metadatas)
        print(f"\nIndexed {len(all_chunks)} total chunks from {len(files)} files into '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    build_index()