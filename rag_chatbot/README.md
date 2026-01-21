# RAG-Based Document Chatbot (FastAPI)

This project is an AI-powered chatbot that answers questions from PDF documents.

The chatbot does not guess answers.
It only responds using the information available inside the documents.


# What this project does

Reads PDF files
Extracts text from documents
Splits text into small chunks
Converts text into embeddings
Stores embeddings in FAISS vector database
Retrieves relevant information based on user questions
Generates answers using Google Gemini
Returns answers with source document name



# Why RAG is used

Large Language Models may generate incorrect answers.

Retrieval-Augmented Generation (RAG) ensures that:
Answers come only from documents
Hallucination is avoided
Responses are reliable and traceable


# Technologies Used
 Python 3.11
 FastAPI
 Google Gemini API
 Sentence Transformers
 FAISS Vector Database
 PyPDF



# Project Structure

rag-chatbot/
│
├── app/
│ ├── main.py
│ ├── loader.py
│ ├── vector_store.py
│ └── rag.py
│
├── data/
│ ├── hospital_admission_policy.pdf
│ └── medical_leave_policy.pdf
│
├── requirements.txt
├── README.md
└── .env



---

# How to Run

# Step 1: Create virtual environment


python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
GEMINI_API_KEY=your_api_key_here
uvicorn app.main:app --reload
http://127.0.0.1:8000/docs
POST /ask

