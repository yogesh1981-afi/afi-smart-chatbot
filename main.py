# main.py (corrected for Render with /ping route)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
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

# Initialize Pinecone using environment variables
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT")
INDEX_NAME = "afi-index"

if not PINECONE_API_KEY or not PINECONE_ENVIRONMENT:
    raise ValueError("Pinecone API Key and Environment must be set as environment variables.")

pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
index = pc.Index(INDEX_NAME)

class EmbedRequest(BaseModel):
    text: str

@app.post("/embed")
async def embed_text(request: EmbedRequest):
    try:
        index_description = pc.describe_index(INDEX_NAME)
        dimension = index_description.dimension

        # Replace with your real model ID in Pinecone if applicable
        embed_response = index.embed(
            vectors=[request.text],
            model="your-embedding-model-id"
        )

        if embed_response and embed_response.vectors:
            return {"embedding": embed_response.vectors[0].values}
        else:
            raise HTTPException(status_code=500, detail="Failed to get embedding from Pinecone.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))