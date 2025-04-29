# AFI Smart Chat Assistant - Embedding API

The **AFI Smart Chat Assistant** Embedding API is designed to generate vector embeddings for user queries and retrieve relevant information from a Pinecone vector database. It uses a HuggingFace model for generating embeddings and Pinecone to search for the top matching knowledge chunks.

### Features:
- Accepts user queries and returns the top 3 relevant knowledge chunks from Pinecone.
- Generates query embeddings using a HuggingFace model.
- Built with **FastAPI** and deployed on **Render**.

### Endpoints:

#### 1. `/query` - Query Search Endpoint

- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "query": "Your question text here"
  }