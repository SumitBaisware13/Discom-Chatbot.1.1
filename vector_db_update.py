import pandas as pd
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer

# Load data from both sources
df1 = pd.read_csv("FINAL_DATA.csv")
df1 = df1.dropna(subset=["Category", "Subcategory", "ISSUE", "Remark_By_Resolver"])
df1 = df1.rename(columns={"ISSUE": "question", "Remark_By_Resolver": "answer"})

df2 = pd.read_csv("estimate.csv")
df2 = df2.dropna(subset=["Query", "Solution"])
df2 = df2.rename(columns={"Query": "question", "Solution": "answer"})

df = pd.concat([df1[["question", "answer"]], df2[["question", "answer"]]], ignore_index=True)

# Generate embeddings and index
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(df["question"].tolist())
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

# Save to pickle
vector_store = {
    "df": df,
    "index": index
}

with open("vector_data.pkl", "wb") as f:
    pickle.dump(vector_store, f)

print("✅ Vector DB updated and saved as vector_data.pkl")
