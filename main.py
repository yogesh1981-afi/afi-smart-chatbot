# main.py (corrected for Render)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pinecone import Pinecone

app = FastAPI()

# Initialize Pinecone
PINECONE_API_KEY = "pcsk_3gacFU_LTFpgKGM7z9jaYVbYSbDYkPCJRJYibAicmzZ4uWQ5Z3Uistcs3G9v1kdz"
PINECONE_ENVIRONMENT = "us-east-1-aws"
INDEX_NAME = "afi-index"

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

class EmbedRequest(BaseModel):
    text: str

@app.post("/embed")
async def embed_text(request: EmbedRequest):
    try:
        # Use Pinecone's serverless embedding
        embed_response = pc.describe_index(INDEX_NAME)
        dimension = embed_response['dimension']

        vector_response = pc.describe_index(INDEX_NAME)['embed']

        # Use the Pinecone native embedding endpoint
        embed_result = pc._sync._api_client._embedding(index_name=INDEX_NAME, input=request.text)

        return {
            "embedding": embed_result["vectors"][0]["values"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "AFI Smart Chat Assistant Embedding API is running 🚀"}