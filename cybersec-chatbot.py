import os, sys
# new imports from langchain_community
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import GPT4All

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# Config
BOOKS_DIR = "F:\\book\\python_pentest_PDFs\\"
INDEX_DIR = "vectorstore"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_PATH   = "F:\\book\\models\\ggml-gpt4all-j-v1.3-groovy.bin"

def load_store():
    if os.path.isdir(INDEX_DIR):
        emb = HuggingFaceEmbeddings(model_name=EMB_MODEL)
        return FAISS.load_local(INDEX_DIR, emb)
    docs = []
    for f in os.listdir(BOOKS_DIR):
        if f.lower().endswith(".pdf"):
            docs += PyPDFLoader(os.path.join(BOOKS_DIR, f)).load()
    chunks = RecursiveCharacterTextSplitter(1000,200).split_documents(docs)
    emb = HuggingFaceEmbeddings(model_name=EMB_MODEL)
    store = FAISS.from_documents(chunks, emb)
    store.save_local(INDEX_DIR)
    return store

vs  = load_store()
llm = GPT4All(model=LLM_PATH, verbose=False)
qa  = RetrievalQA.from_chain_type(llm=llm, retriever=vs.as_retriever({"k":5}))

def chat():
    print("Cybersec Chatbot (type 'exit' to quit)")
    while True:
        q = input("You> ").strip()
        if q.lower() in ("exit","quit"):
            print("Goodbye!")
            break
        print("Bot> "+qa.run(q))

if __name__=="__main__":
    if not os.path.isdir(BOOKS_DIR):
        print(f"Please add PDFs into '{BOOKS_DIR}' and rerun.")
        sys.exit(1)
    chat()
