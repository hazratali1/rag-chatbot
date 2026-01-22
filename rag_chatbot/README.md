# RAG-Based Document Chatbot (FastAPI)

This project is an AI-powered chatbot that answers questions from PDF documents.

The chatbot does not guess answers.
It only responds using the information available inside the uploaded documents.

## What this project does

* Reads PDF files
* Extracts text from documents
* Splits text into small chunks
* Converts text into embeddings
* Stores embeddings in FAISS vector database
* Searches relevant information based on user questions
* Generates answers using Google Gemini
* Returns answers along with source document name

## Why RAG is used

Large Language Models can generate incorrect or imaginary answers.

Retrieval-Augmented Generation (RAG) ensures that:

* Answers come only from the documents
* Hallucination is avoided
* Output is reliable and traceable
* Source documents can be identified

## Technologies Used

* Python 3.11
* FastAPI
* Google Gemini API
* Sentence Transformers
* FAISS Vector Database
* PyPDF
* 
## Project Structure

```
rag-chatbot/
│
├── app/
│   ├── main.py
│   ├── loader.py
│   ├── vector_store.py
│   └── rag.py
│
├── data/
│   ├── hospital_admission_policy.pdf
│   └── medical_leave_policy.pdf
│
├── requirements.txt
├── README.md
└── .env
```

## How the system works

```
PDF Files
   ↓
Text Extraction
   ↓
Text Chunking
   ↓
Embeddings Creation
   ↓
FAISS Vector Database
   ↓
Relevant Chunk Retrieval
   ↓
Gemini LLM
   ↓
Final Answer
```

## API Endpoints

### Health Check

```
GET /health
```

Response:

```json
{
  "status": "running"
}
```

### Ask Question (RAG)

```
POST /ask
```

Example:

```
POST /ask?question=What is the hospital admission policy?
```

Response:

```json
{
  "answer": "...",
  "source": "hospital_admission_policy.pdf"
}
```
## How to Run

### Step 1: Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Add environment variable

Create a `.env` file: (alredy exist in folder )

```env
GEMINI_API_KEY=your_api_key_here
```
### Step 4: Run the application

```bash
uvicorn app.main:app --reload
```
### Open API Docs

```
http://127.0.0.1:8000/docs
```
## Notes

* The chatbot does not answer outside document context
* If information is not present, the model responds accordingly
* FAISS is used for fast semantic search
* Suitable for company documents, policies, manuals, and reports
  
## Future Improvements

* Advanced retrievers (MMR)
* Hybrid search (keyword + vector)
* Chat history memory
* Docker deployment
* UI integration



