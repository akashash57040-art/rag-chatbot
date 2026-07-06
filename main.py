"""
main.py
FastAPI server exposing:
  POST /chat    -> ask a question, get an answer + sources
  POST /ingest  -> rebuild the vector index from ./documents
  GET  /        -> serves the chat UI (static/index.html)
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag import generate_answer
from ingest import build_index

app = FastAPI(title="RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []


@app.post("/chat")
def chat(req: ChatRequest):
    history = [{"role": m.role, "content": m.content} for m in req.history]
    result = generate_answer(req.message, history)
    return result


@app.post("/ingest")
def ingest():
    build_index()
    return {"status": "index rebuilt"}


# Serve the frontend at the root
app.mount("/", StaticFiles(directory="static", html=True), name="static")