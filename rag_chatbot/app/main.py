from fastapi import FastAPI
from dotenv import load_dotenv
import os

from app.loader import load_and_split_pdfs
from app.vector_store import create_vector_store
from app.rag import ask_question

load_dotenv()

app = FastAPI(title="RAG Chatbot API")

chunks = load_and_split_pdfs()

texts = [c.page_content for c in chunks]
sources = [c.metadata["source"] for c in chunks]

vectorstore = create_vector_store(chunks)
embeddings = vectorstore.index.reconstruct_n(0, vectorstore.index.ntotal)


@app.get("/health")
def health():
    return {"status": "running"}


@app.post("/ask")
def ask(question: str):
    result = ask_question(question, texts, embeddings, sources)
    return result
