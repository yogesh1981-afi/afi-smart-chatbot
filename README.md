# AFI Smart Chat Assistant - Embedding API

This API accepts a user query, generates an embedding using HuggingFace model, searches Pinecone vector database, and returns top 3 matching knowledge chunks.

- Endpoint: `/query`
- Method: `POST`
- Body:
  ```json
  {
    "query": "Your question text here"
  }