from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import os

app = FastAPI()

# Global variables to hold model and index (initialized on startup)
model = None
index = None

# Load environment variables
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
INDEX_NAME = "afi-index"

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY environment variable must be set.")

# --- FastAPI Startup Event ---
@app.on_event("startup")
async def startup_event():
    global model, index
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(INDEX_NAME)
        print(f"✅ Connected to Pinecone index: {INDEX_NAME}")
    except Exception as e:
        print(f"❌ Pinecone initialization failed: {e}")
        raise RuntimeError(f"❌ Failed to connect to Pinecone index: {e}")

    try:
        model_id = 'sentence-transformers/all-MiniLM-L6-v2'
        model = SentenceTransformer(model_id)
        print(f"✅ Loaded model: {model_id}")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        raise RuntimeError(f"❌ Failed to load model: {e}")

# Health check
@app.get("/ping")
async def ping():
    return {"message": "Embedding API is live"}

# Root
@app.get("/")
async def root():
    return {"message": "AFI Smart Chat Assistant Embedding API is running 🚀"}

# Request and response models
class EmbedRequest(BaseModel):
    text: str

class EmbedResponse(BaseModel):
    embedding: list[float]

# Embed endpoint
@app.post("/embed", response_model=EmbedResponse)
async def embed_text(request: EmbedRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")
    try:
        embedding = model.encode(request.text).tolist()
        return {"embedding": embedding}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate embedding: {e}")