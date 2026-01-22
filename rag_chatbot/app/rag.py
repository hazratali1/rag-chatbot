"""
RAG implementations.

- basic: manual cosine similarity (I TOOK TWO SMALL PDF)
- advanced: FAISS retriever (WHEN WE WILL USE A BIG DOCUMENT)
"""

import os
import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))



# BASIC RAG (I RUN THIS FOR SAMLL TAKEN PDF)


_basic_model = SentenceTransformer("all-MiniLM-L6-v2")


def _cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def ask_basic(question, texts, embeddings, sources):
    q_emb = _basic_model.encode(question)

    scores = [_cosine(q_emb, e) for e in embeddings]
    idx = int(np.argmax(scores))

    context = texts[idx]

    prompt = f"""
Answer strictly from the context.

Context:
{context}

Question:
{question}
"""

    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)

    return {
        "answer": response.text,
        "source": sources[idx],
        "mode": "basic"
    }



# ADVANCED RAG (IF TAKEN PDF IS BIG USE FAISS CAN BE SERCH)


def ask_advanced(question, vectorstore):
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    docs = retriever.get_relevant_documents(question)

    context = "\n\n".join(d.page_content for d in docs)

    prompt = f"""
You are an AI assistant.

Use only the provided context.
If the answer is not present, reply:
"I don't know."

Context:
{context}

Question:
{question}
"""

    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)

    return {
        "answer": response.text,
        "sources": [d.metadata.get("source") for d in docs],
        "mode": "advanced"
    }
