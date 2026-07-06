"""
rag.py
Core retrieval-augmented generation logic:
  1. Embed the user's query and retrieve top-k relevant chunks from Chroma
  2. Build a prompt that injects those chunks as context
  3. Call a local Ollama model with that prompt + conversation history
  4. Return the answer along with the sources used (for citations)
"""

import chromadb
import requests

DB_DIR = "chroma_db"
COLLECTION_NAME = "docs"
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.1"
TOP_K = 4

client = chromadb.PersistentClient(path=DB_DIR)


def get_collection():
    return client.get_collection(COLLECTION_NAME)


def retrieve(query: str, k: int = TOP_K):
    """
    Embed the query (Chroma does this automatically with the same
    embedding function used at ingest time) and return the top-k chunks.
    """
    collection = get_collection()
    results = collection.query(query_texts=[query], n_results=k)

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        chunks.append({"text": doc, "source": meta["source"], "chunk_index": meta["chunk_index"], "distance": dist})
    return chunks


def build_system_prompt(chunks: list[dict]) -> str:
    """Inject retrieved chunks as labeled context blocks the model can cite."""
    if not chunks:
        context_block = "No relevant context was found in the knowledge base."
    else:
        context_block = "\n\n".join(
            f"[Source: {c['source']}, chunk {c['chunk_index']}]\n{c['text']}" for c in chunks
        )

    return f"""You are a helpful assistant that answers questions using ONLY the context provided below.

- If the answer isn't contained in the context, say you don't have enough information — do not make things up.
- When you use information from the context, mention which source it came from, e.g. "(Source: filename.pdf)".
- Keep answers concise and directly relevant to the question.

CONTEXT:
{context_block}
"""


def generate_answer(query: str, history: list[dict]) -> dict:
    """
    history: list of {"role": "user"|"assistant", "content": str}, most recent last.
    Returns: {"answer": str, "sources": [...]}
    """
    chunks = retrieve(query)
    system_prompt = build_system_prompt(chunks)

    # Ollama's /api/chat expects the system prompt as a normal message in the list
    messages = [{"role": "system", "content": system_prompt}] + history + [
        {"role": "user", "content": query}
    ]

    response = requests.post(
        OLLAMA_URL,
        json={"model": MODEL, "messages": messages, "stream": False},
        timeout=120,
    )
    response.raise_for_status()
    answer_text = response.json()["message"]["content"]

    # De-duplicate sources for a clean citation list
    seen = set()
    sources = []
    for c in chunks:
        if c["source"] not in seen:
            seen.add(c["source"])
            sources.append(c["source"])

    return {"answer": answer_text, "sources": sources}