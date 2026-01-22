# FastAPI

from fastapi import FastAPI
from dotenv import load_dotenv
import os

from app.loader import load_and_split_pdfs
from app.vector_store import create_vector_store
from app.rag import (
    ask_question_basic,
    ask_question_advanced
)

load_dotenv()

app = FastAPI(title="RAG Chatbot API")

# LOAD PDF FILES

chunks = load_and_split_pdfs()

texts = [c.page_content for c in chunks]
sources = [c.metadata["source"] for c in chunks]

# VECTOR STORE
vectorstore = create_vector_store(chunks)

# embeddings used only for BASIC RAG
embeddings = vectorstore.index.reconstruct_n(
    0,
    vectorstore.index.ntotal
)

@app.get("/health")
def health():
    return {"status": "running"}

@app.post("/ask")
def ask_basic(question: str):
    """
    BASIC RAG
    - manual cosine similarity
    - suitable for small PDFs
    """
    return ask_question_basic(
        question,
        texts,
        embeddings,
        sources
    )



# ADVANCED RAG (FAISS RETRIEVER)


@app.post("/ask/advanced")
def ask_advanced(question: str):
    """
    ADVANCED RAG
    - FAISS retriever
    - scalable for large PDFs
    """
    return ask_question_advanced(
        question,
        vectorstore
    )
