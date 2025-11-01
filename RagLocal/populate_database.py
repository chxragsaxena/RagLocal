import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from get_embedding_function import get_embedding_function

DATA_DIR = "data"
CHROMA_DIR = "chroma"

def load_documents_from_folder(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
            print(f"Loaded: {filename}")
    return documents

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)

def create_vectorstore(docs, persist_dir, embedding_func):
    db = Chroma.from_documents(
        documents=docs,
        embedding=embedding_func,
        persist_directory=persist_dir
    )
    db.persist()
    print(f"✅ Vectorstore created at '{persist_dir}' with {len(docs)} chunks.")

def reset_vectorstore(persist_dir):
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
        print("🧹 Cleared previous vectorstore.")

def main(reset=False):
    if reset:
        reset_vectorstore(CHROMA_DIR)

    documents = load_documents_from_folder(DATA_DIR)
    if not documents:
        print("⚠️ No PDF files found in 'data/' folder.")
        return

    split_docs = split_documents(documents)
    embedding_func = get_embedding_function()
    create_vectorstore(split_docs, CHROMA_DIR, embedding_func)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest PDF files into Chroma vectorstore.")
    parser.add_argument("--reset", action="store_true", help="Clear existing vectorstore before populating.")
    args = parser.parse_args()

    main(reset=args.reset)
