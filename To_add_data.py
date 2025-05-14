import os
import pandas as pd
import numpy as np
import faiss
import pickle
import argparse
from sentence_transformers import SentenceTransformer

def try_read_and_format(file_path):
    try:
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file_path)
        else:
            return None

        df.columns = df.columns.str.strip().str.lower()

        if "query" in df.columns and "solution" in df.columns:
            df = df.rename(columns={"query": "question", "solution": "answer"})
        elif "question" in df.columns and "answer" in df.columns:
            pass
        else:
            return None

        df = df.dropna(subset=["question", "answer"])
        return df[["question", "answer"]]

    except Exception as e:
        print(f"⚠️ Failed to load file {file_path}: {e}")
        return None

def append_to_vector_store(pkl_path="vector_data.pkl", upload_path=None):
    if not os.path.exists(pkl_path):
        print(f"❌ Vector DB file '{pkl_path}' not found.")
        return

    with open(pkl_path, "rb") as f:
        vector_store = pickle.load(f)

    df_existing = vector_store["df"]
    index = vector_store["index"]

    # Ask user to provide file path if not provided
    if not upload_path:
        upload_path = input("📁 Enter path to your Q&A file (CSV/XLSX): ").strip()

    new_df = try_read_and_format(upload_path)
    if new_df is None or new_df.empty:
        print("ℹ️ No valid data found in the file.")
        return

    # Remove duplicates
    new_df = new_df[~new_df["question"].isin(df_existing["question"])]
    if new_df.empty:
        print("✅ No new unique questions found to add.")
        return

    print(f"➕ Adding {len(new_df)} new Q&A pairs...")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    new_embeddings = model.encode(new_df["question"].tolist())
    index.add(np.array(new_embeddings))

    updated_df = pd.concat([df_existing, new_df], ignore_index=True)

    with open(pkl_path, "wb") as f:
        pickle.dump({"df": updated_df, "index": index}, f)

    print("✅ Vector DB updated with new entries!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Append Q&A to vector DB")
    parser.add_argument("--file", type=str, help="Path to Q&A file to upload (CSV/XLSX)")
    args = parser.parse_args()

    append_to_vector_store(upload_path=args.file)
