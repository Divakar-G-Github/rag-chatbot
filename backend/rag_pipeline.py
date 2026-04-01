import os
import chromadb
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

load_dotenv()

# ---- Global in-memory client ----
chroma_client = chromadb.EphemeralClient()
collection_name = "rag_collection"

# ---- Lightweight built-in embedding ----
embedding_function = ONNXMiniLM_L6_V2()

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

# ---- STEP 3: Store in memory ----
def create_vectorstore(chunks):
    try:
        chroma_client.delete_collection(collection_name)
    except:
        pass

    collection = chroma_client.create_collection(
        name=collection_name,
        embedding_function=embedding_function
    )

    # Add chunks directly to chromadb
    documents = [chunk.page_content for chunk in chunks]
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    collection.add(
        documents=documents,
        ids=ids
    )
    print(f"✅ Added {len(chunks)} chunks to vectorstore")
    return collection

# ---- STEP 4: Get answer ----
def get_answer(question: str):
    collection = chroma_client.get_collection(
        name=collection_name,
        embedding_function=embedding_function
    )

    # Query top 3 relevant chunks
    results = collection.query(
        query_texts=[question],
        n_results=3
    )

    docs = results["documents"][0]
    context = "\n\n".join(docs)

    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )

    prompt = f"""Answer the question based only on the context below.
If you don't know the answer from the context, say "I don't know".

Context:
{context}

Question: {question}

Answer:"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "sources": docs
    }

# ---- FULL PIPELINE ----
def process_pdf(file_path: str):
    documents = load_pdf(file_path)
    chunks = split_documents(documents)
    create_vectorstore(chunks)
    return f"Successfully processed {len(chunks)} chunks from your document"