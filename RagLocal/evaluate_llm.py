import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import Ollama, OllamaEmbeddings
from langchain.chains import RetrievalQA

# 📂 Load all PDF files from 'data' folder
data_dir = "data"
pdf_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pdf")]

all_docs = []

print(f"📄 Found {len(pdf_files)} PDF(s) in {data_dir}")

for pdf_path in pdf_files:
    print(f"📑 Loading: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    all_docs.extend(docs)

# 📖 Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(all_docs)

# 🧠 Create vector store with Ollama embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

# 🔍 Load LLM and QA chain
llm = Ollama(model="llama3")
retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# 🧠 Ask questions in a loop
while True:
    query = input("\n🔍 Ask a question (or type 'exit' to quit): ")
    if query.lower() == "exit":
        break

    result = qa_chain.invoke({"query": query})
    print(f"\n🧠 Answer: {result['result']}")
