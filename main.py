from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import pinecone
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

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
index = pinecone.Index(INDEX_NAME)

# Load the Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')  # You can replace this with your preferred model

class EmbedRequest(BaseModel):
    text: str

@app.post("/embed")
async def embed_text(request: EmbedRequest):
    try:
        # Generate embedding using SentenceTransformer
        embedding = model.encode(request.text).tolist()  # Convert the embedding to a list for Pinecone

        # Upsert the embedding into Pinecone
        upsert_response = index.upsert(
            vectors=[("unique-id", embedding)],  # Use a unique ID for each embedding
        )

        if upsert_response:
            return {"embedding": embedding}

        else:
            raise HTTPException(status_code=500, detail="Failed to upsert embedding into Pinecone.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))