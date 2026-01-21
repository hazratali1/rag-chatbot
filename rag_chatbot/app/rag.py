#RAG logic

import os
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import numpy as np


genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = SentenceTransformer("all-MiniLM-L6-v2")


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def ask_question(question, texts, embeddings, sources):
    question_embedding = model.encode(question)

    scores = [cosine_similarity(question_embedding, emb) for emb in embeddings]
    top_index = int(np.argmax(scores))

    context = texts[top_index]

    prompt = f"""
Answer ONLY from the context below.
If not found, say "Information not available in the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""

    gemini = genai.GenerativeModel("gemini-pro")
    response = gemini.generate_content(prompt)

    return {
        "answer": response.text,
        "source": sources[top_index]
    }
