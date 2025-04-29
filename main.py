from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import os

app = FastAPI()

# Health check route
@app.get("/ping")
def ping():
    return {"message": "Embedding API is live"}

# Root route
@app.get("/")
def read_root():
    return {"message": "AFI Smart Chat Assistant Embedding API is running 🚀"}

# Load environment variables
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT")
INDEX_NAME = "afi-index"

if not PINECONE_API_KEY or not PINECONE_ENVIRONMENT:
    raise ValueError("Pinecone API Key and Environment must be set as environment variables.")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Load SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Request model
class EmbedRequest(BaseModel):
    text: str

# Embed endpoint
@app.post("/embed")
async def embed_text(request: EmbedRequest):
    try:
        # Generate embedding
        embedding = model.encode(request.text).tolist()

        # Upsert to Pinecone with a dummy ID
        upsert_response = index.upsert(
            vectors=[("unique-id", embedding)]
        )

        if upsert_response:
            return {"embedding": embedding}
        else:
            raise HTTPException(status_code=500, detail="Upsert failed.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))