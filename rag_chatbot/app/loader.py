
#PDF chunks

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os


def load_and_split_pdfs(data_folder="data"):
    documents = []

    for file in os.listdir(data_folder):
        if file.endswith(".pdf"):
            file_path = os.path.join(data_folder, file)

            loader = PyPDFLoader(file_path)
            docs = loader.load()

            for d in docs:
                d.metadata["source"] = file

            documents.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    return chunks
