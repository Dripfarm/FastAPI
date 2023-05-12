from fastapi import FastAPI
from typing import List
import numpy as np
import faiss

app = FastAPI()

# Load embeddings from a local file
embeddings_np = np.load('embeddings.npy')

# Load the sentences from the file
with open('rizzes.txt', 'r') as f:
    sentences = [line.strip() for line in f]

# Build the index
index = faiss.IndexFlatL2(embeddings_np.shape[1])
index.add(embeddings_np)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/recommend/{rizzIndex}")
def recommend(rizzIndex: int):
    D, I = index.search(embeddings_np[rizzIndex:rizzIndex+1], k=4)
    recommendations = [{"index": int(i), "text": sentences[i]} for i in I[0]]
    return {"recommendations": recommendations}
    
@app.get("/compliments")
def get_compliments():
    with open('rizzes.txt', 'r') as f:
        compliments = [line.strip() for line in f]
    return {"compliments": compliments}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
