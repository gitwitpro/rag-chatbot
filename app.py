from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

documents = []
sources = []

# Load all files
data_path = "data"
for file in os.listdir(data_path):
    with open(os.path.join(data_path, file), "r") as f:
        for line in f.readlines():
          clean_line = line.strip()
          if clean_line:  # skip empty lines
            documents.append(clean_line)
            sources.append(file)

# Create embeddings
doc_embeddings = model.encode(documents)

# FAISS index
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
faiss.normalize_L2(doc_embeddings)
index.add(np.array(doc_embeddings))

def search(query, k=3):
    query_embedding = model.encode([query])
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding, k)

    results = []
    for idx, i in enumerate(indices[0]):
      if i < len(documents):
        score = distances[0][idx]
        results.append((documents[i], sources[i], score))

    return results

# Main loop
while True:
    query = input("\nAsk your question (type 'exit' to quit): ")

    if query.lower() == "exit":
        print("Goodbye!")
        break

    results = search(query)
    if not results:
      print("❌ No relevant data found.")
      continue

    print("\n" + "="*50)
    print("🔍 Retrieved Context:\n")

    for text, src, score in results:
      print(f"• {text} ({src}) [score: {score:.4f}]")

    print("\n" + "="*50)
    print("💡 Final Answer:\n")

    best_answer = results[0][0]
    print(f"👉 {best_answer}")

    print("\n" + "="*50)
