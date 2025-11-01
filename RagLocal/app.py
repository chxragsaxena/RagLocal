import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA

# 🗂 Load PDFs 
def load_documents_from_folder(folder_path):
    docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder_path, filename))
            docs.extend(loader.load())
    return docs

# 🔧 Initialize RAG 
@st.cache_resource
def setup_rag():
    raw_docs = load_documents_from_folder("data")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(raw_docs)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectordb = Chroma.from_documents(chunks, embedding=embeddings)

    retriever = vectordb.as_retriever()
    llm = ChatOllama(model="llama3")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain

qa_chain = setup_rag()

# 🌐 Streamlit UI
st.set_page_config(page_title="Private PDF Chat", layout="wide")
st.title("🤖 Ask Anything About Your PDF Files")

user_question = st.text_input("🔍 Your question:", placeholder="e.g. What is Jio Platforms?")

if user_question:
    with st.spinner("Thinking..."):
        response = qa_chain.run(user_question)
        st.markdown(f"**🧠 Answer:** {response}")
