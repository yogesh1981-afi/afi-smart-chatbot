from fastapi import FastAPI, Request
import pinecone
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

app = FastAPI()

# Pinecone API credentials
PINECONE_API_KEY = "pcsk_3gacFU_LTFpgKGM7z9jaYVbYSbDYkPCJRJYib2sUHibAicmzZ4uWQ5Z3Uistcs3G9v1kdz"
PINECONE_ENVIRONMENT = "us-east-1-aws"

# Initialize Pinecone client
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("afi-index")

# Load HuggingFace embedding model
tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")
model = AutoModel.from_pretrained("intfloat/multilingual-e5-large")

# Helper function for embedding
def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        model_output = model(**inputs)
        pooled = average_pool(model_output.last_hidden_state, inputs["attention_mask"])
        embedding = F.normalize(pooled, p=2, dim=1)
    return embedding.squeeze(0).cpu().tolist()

def average_pool(last_hidden_states, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_states.size())
    masked_embeddings = last_hidden_states * mask
    summed = masked_embeddings.sum(dim=1)
    counts = mask.sum(dim=1)
    mean_pooled = summed / counts
    return mean_pooled

# API Endpoint
@app.post("/query")
async def query_api(request: Request):
    data = await request.json()
    query_text = data.get("query", "")
    if not query_text:
        return {"error": "No query text provided."}

    query_vector = embed_text(query_text)
    search_result = index.query(vector=query_vector, top_k=3, include_metadata=True)

    answers = []
    for match in search_result['matches']:
        answers.append({
            "score": match['score'],
            "text": match['metadata'].get('text', '')
        })

    return {"answers": answers}