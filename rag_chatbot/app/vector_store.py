# Creates vector database

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def create_vector_store(chunks):
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_documents(
        documents=chunks,
        embedding=embedding
    )
