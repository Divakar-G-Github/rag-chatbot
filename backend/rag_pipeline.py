import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma
import chromadb

load_dotenv()

# ---- Global in-memory client ----
chroma_client = chromadb.EphemeralClient()   # ← stays in RAM, no disk needed
collection_name = "rag_collection"

# ---- STEP 1: Load PDF ----
def load_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages")
    return documents

# ---- STEP 2: Split into chunks ----
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks

# ---- STEP 3: Create embeddings & store in memory ----
def create_vectorstore(chunks):
    embeddings = FastEmbedEmbeddings()

    # Delete existing collection if exists
    try:
        chroma_client.delete_collection(collection_name)
    except:
        pass

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        client=chroma_client,               # ← in-memory client
        collection_name=collection_name
    )
    print("Vectorstore created in memory")
    return vectorstore

# ---- STEP 4: Load existing vectorstore from memory ----
def load_vectorstore():
    embeddings = FastEmbedEmbeddings()
    vectorstore = Chroma(
        client=chroma_client,               # ← in-memory client
        collection_name=collection_name,
        embedding_function=embeddings
    )
    return vectorstore

# ---- STEP 5: Answer question using RAG ----
def get_answer(question: str):
    vectorstore = load_vectorstore()

    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""Answer the question based only on the context below.
If you don't know the answer from the context, say "I don't know".

Context:
{context}

Question: {question}

Answer:"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "sources": [doc.page_content for doc in docs]
    }

# ---- FULL PIPELINE: Process uploaded PDF ----
def process_pdf(file_path: str):
    documents = load_pdf(file_path)
    chunks = split_documents(documents)
    create_vectorstore(chunks)
    return f"Successfully processed {len(chunks)} chunks from your document"