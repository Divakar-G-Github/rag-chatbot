# 🤖 RAG Document Chatbot

A production-ready RAG (Retrieval Augmented Generation) chatbot 
that answers questions from uploaded PDF documents.

## 🛠️ Tech Stack
- **LLM:** Llama 3 via Groq API (Free)
- **Embeddings:** HuggingFace sentence-transformers
- **Vector DB:** ChromaDB
- **Framework:** LangChain
- **Backend:** FastAPI
- **Frontend:** Streamlit
- **Container:** Docker

## ⚙️ How It Works
1. Upload any PDF document
2. Text is extracted and split into chunks
3. Chunks converted to embeddings and stored in ChromaDB
4. Ask questions in natural language
5. Relevant chunks retrieved and sent to Llama 3
6. Answer returned with source references

## 🚀 How to Run
1. Clone the repo
2. Create `.env` file with your `GROQ_API_KEY`
3. Install dependencies: `pip install -r backend/requirements.txt`
4. Run backend: `uvicorn main:app --reload --port 8000`
5. Run frontend: `streamlit run frontend/app.py`

## 📸 Screenshots
(Add your screenshots here)
