# build_rag_index.py

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os

# Ensure rag folder exists
if not os.path.exists("rag"):
    os.makedirs("rag")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Check guidelines file
if not os.path.exists("rag/guidelines.txt"):
    raise FileNotFoundError("rag/guidelines.txt not found!")

with open("rag/guidelines.txt", "r") as f:
    docs = f.read().split("\n\n")

embeddings = model.encode(docs)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

with open("rag/vector_store.pkl", "wb") as f:
    pickle.dump((docs, index), f)

print("✅ RAG Index built successfully")