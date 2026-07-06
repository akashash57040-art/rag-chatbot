# 🤖 RAG Chatbot — Chat With Your Own Documents

A Retrieval-Augmented Generation (RAG) chatbot that answers questions grounded in your own documents, with source citations — not hallucinated guesses. Runs **fully free and offline** using a local LLM through [Ollama](https://ollama.com), with no API keys or usage costs.

![Python](https://img.shields.io/badge/python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-backend-teal)
![Chroma](https://img.shields.io/badge/ChromaDB-vector%20store-orange)
![Ollama](https://img.shields.io/badge/Ollama-local%20LLM-black)

## Why this project

Most chatbot demos just wrap an LLM API and call it done. This one implements the full RAG pipeline from scratch — chunking strategy, vector retrieval, prompt grounding, and citation tracing — so answers are backed by real source material instead of the model's memory.
## Features

- 📄 **Multi-format ingestion** — reads `.txt`, `.md`, and `.pdf` files
- ✂️ **Smart chunking** — splits on sentence boundaries with overlap, instead of blindly cutting every N characters
- 🔍 **Semantic search** — retrieves the most relevant chunks via vector similarity (ChromaDB, local embeddings)
- 📌 **Source citations** — every answer references which document it came from
- 💬 **Simple chat UI** — clean, dark-themed interface, no framework bloat
- 🆓 **Zero cost** — runs entirely locally via Ollama; no OpenAI/Anthropic API key required
- 🔌 **Swappable backend** — one function call away from using Claude/GPT instead of a local model, if you want higher-quality answers later

## Tech stack

| Layer | Tool |
|---|---|
| Backend | FastAPI |
| Vector database | ChromaDB (persistent, local) |
| Embeddings | `all-MiniLM-L6-v2` (auto-downloaded by Chroma) |
| LLM | Ollama (`llama3.1`, swappable) |
| Frontend | Vanilla HTML/CSS/JS |
| PDF parsing | pypdf |

## Getting started

### Prerequisites
- Python 3.11 ([download](https://www.python.org/downloads/))
- [Ollama](https://ollama.com) installed and running

### Setup

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/rag-chatbot.git
cd rag-chatbot

# 2. Pull a local model (one-time download)
ollama pull llama3.1

# 3. Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 4. Install dependencies
pip install -r requirements.txt

# 5. Add your documents to documents/ (.txt, .md, .pdf)

# 6. Build the vector index
python ingest.py

# 7. Start the server
uvicorn main:app --reload --port 8000
```

Open **http://localhost:8000** and start asking questions.

## Project structure
## How the RAG pipeline works

1. **Ingestion** (`ingest.py`) — documents are split into ~800-character chunks with 150-character overlap, breaking on sentence boundaries where possible so context isn't cut mid-thought. Each chunk is embedded and stored in ChromaDB alongside metadata (source file, chunk index).
2. **Retrieval** (`rag.py`) — the user's question is embedded using the same model, and the top-4 most similar chunks are retrieved via cosine similarity.
3. **Grounded generation** — retrieved chunks are injected into a system prompt instructing the model to answer *only* from that context, and to say so explicitly if the answer isn't present — reducing hallucination.
4. **Citations** — the source filename for each chunk used is returned alongside the answer and displayed in the UI.

## Roadmap / possible extensions

- [ ] Hybrid search (BM25 + embeddings) for exact-match queries (names, numbers)
- [ ] Re-ranking retrieved chunks with a cross-encoder before generation
- [ ] Evaluation harness — a labeled Q&A test set to measure retrieval accuracy
- [ ] Streaming responses (token-by-token, like ChatGPT)
- [ ] Swap ChromaDB for a hosted vector DB (Pinecone/Weaviate) for scale
- [ ] Docker deployment

## License

MIT — free to use, modify, and build on.