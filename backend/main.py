import os
import shutil
import traceback
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

if os.path.exists("vectorstore"):
    shutil.rmtree("vectorstore")
    os.makedirs("vectorstore")
    print("✅ Cleared old vectorstore")

app = FastAPI(title="RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ---- Import RAG pipeline safely ----
try:
    from rag_pipeline import process_pdf, get_answer
    print("✅ RAG pipeline imported successfully")
except Exception as e:
    print(f"❌ Failed to import rag_pipeline: {e}")
    traceback.print_exc()

# ---- Health Check ----
@app.get("/")
def root():
    return {"status": "RAG Chatbot API is running"}

# ---- Upload PDF ----
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        os.makedirs("data", exist_ok=True)
        file_path = f"data/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        result = process_pdf(file_path)
        return {"message": result, "filename": file.filename}
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}

# ---- Ask Question ----
class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        result = get_answer(request.question)
        return result
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}