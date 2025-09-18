# rag_build.py
import os, pickle
import numpy as np
import faiss
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import cohere

# Here's where environment variables are loaded
load_dotenv(".env")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
if not COHERE_API_KEY:
    raise ValueError("❌ COHERE_API_KEY missing in .env")

co = cohere.Client(COHERE_API_KEY)
EMBED_MODEL = "embed-english-v3.0"

PDF_FOLDER = "data/cvs"
INDEX_PATH = "faiss_index"
METADATA_PATH = "faiss_metadata.pkl"

def extract_text_from_pdfs(folder):
    chunks, metadata = [], []
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".pdf"):
            continue
        path = os.path.join(folder, fname)
        try:
            reader = PdfReader(path)
            full_text = "\n".join(page.extract_text() or "" for page in reader.pages)
            # This chunks text into roughly 1500 characters
            chunk_size = 1500
            for i in range(0, len(full_text), chunk_size):
                chunk = full_text[i:i+chunk_size].strip()
                if chunk:
                    chunks.append(chunk)
                    metadata.append({"filename": fname, "text": chunk})
            print(f"📄 Processed {fname} -> {len(full_text)} chars, {len(metadata)} chunks total")
        except Exception as e:
            print(f"❌ Failed to process {fname}: {e}")
    return chunks, metadata

def embed_texts(texts):
    all_embeddings = []
    for i in range(0, len(texts), 90):  # this sets batch size for Cohere
        batch = texts[i:i+90]
        resp = co.embed(texts=batch, model=EMBED_MODEL, input_type="search_document")
        for emb in resp.embeddings:
            v = np.array(emb, dtype="float32")
            v /= np.linalg.norm(v) + 1e-12  # normalizing like crazy
            all_embeddings.append(v)
        print(f"🧠 Embedded {len(all_embeddings)}/{len(texts)} chunks...")
    return np.vstack(all_embeddings)

def main():
    print("📄 Extracting text from PDFs...")
    chunks, metadata = extract_text_from_pdfs(PDF_FOLDER)
    if not chunks:
        raise ValueError("No text extracted. Check your PDFs in data/cvs/")
    print(f"✅ Total chunks: {len(chunks)}")

    print(f"🧠 Embedding {len(chunks)} chunks with Cohere ({EMBED_MODEL})...")
    X = embed_texts(chunks)

    print("🔹 Building FAISS index...")
    dim = X.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(X)

    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)
    faiss.write_index(index, INDEX_PATH)

    print(f"✅ FAISS index built and saved ({len(metadata)} chunks, dim={dim})")

if __name__ == "__main__":
    main()
