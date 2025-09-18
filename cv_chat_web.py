# cv_chat_web.py
from flask import Flask, render_template, request
import os, pickle, traceback
import numpy as np, faiss
from dotenv import load_dotenv
from openai import OpenAI
import cohere

# Loading environment variables
load_dotenv(dotenv_path=".env")

# Here are the API Keys, a real pain in the ass if you don't load all the resources on your virtual environement, check requirements.txt
OR_KEY = os.getenv("OPENROUTER_API_KEY")
if not OR_KEY:
    raise ValueError("❌ OPENROUTER_API_KEY missing in .env")
client = OpenAI(api_key=OR_KEY, base_url="https://openrouter.ai/api/v1")

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
if not COHERE_API_KEY:
    raise ValueError("❌ COHERE_API_KEY missing in .env")
co = cohere.Client(COHERE_API_KEY)
EMBED_MODEL = "embed-english-v3.0"

# Paths used
INDEX_PATH = "faiss_index"
METADATA_PATH = "faiss_metadata.pkl"

print("🔹 Loading FAISS index...")
if not os.path.exists(INDEX_PATH):
    raise FileNotFoundError(f"{INDEX_PATH} not found. Run rag_build.py first.")
index = faiss.read_index(INDEX_PATH)

if not os.path.exists(METADATA_PATH):
    raise FileNotFoundError(f"{METADATA_PATH} not found. Run build_metadata.py first.")
with open(METADATA_PATH, "rb") as f:
    metadata = pickle.load(f)

texts = [m["text"] for m in metadata]
filenames = [m["filename"] for m in metadata]

# Flask app 
# Flask is used here as a lightweight web server to:
# 1. Serve the chat interface (index.html) to the browser
# 2. Receive user questions via POST requests
# 3. Run embeddings + FAISS search + LLM reasoning
# 4. Send the generated answer back to the browser for display
app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

# Helper functions

# Ensures the context passed to the LLM does not exceed token limits.
# If the string is too long, it is truncated and a notice is added.
def safe_truncate(s: str, max_chars: int = 12000) -> str:
    return s if len(s) <= max_chars else s[:max_chars] + "\n\n[Context truncated]\n"

# Converts the user's question into a dense vector (embedding) using Cohere.
#Steps:
   # - Use Cohere's embed endpoint with input_type="search_query"
   # - Normalize the vector to unit length (important for cosine similarity)
   # - Return as a 2D NumPy array for FAISS to use
def embed_query(q: str) -> np.ndarray:
    q2 = q.strip() if q else " "
    resp = co.embed(texts=[q2], model=EMBED_MODEL, input_type="search_query")
    v = np.asarray(resp.embeddings[0], dtype="float32")
    v /= (np.linalg.norm(v) + 1e-12)
    return v.reshape(1, -1)

# Performs semantic search on the FAISS index to find the most relevant CV chunks.
   # - Embeds the user query
   # - Searches FAISS for the nearest neighbors (overfetching to deduplicate per CV)
   # - Returns a list of (filename, text) tuples for the top_k unique CVs
def query_cv(question: str, top_k: int = 8):
    try:
        qv = embed_query(question)
        distances, indices = index.search(qv, top_k * 3)  # Overfetch to dedupe
        results, seen = [], set()
        for idx in indices[0]:
            if 0 <= idx < len(filenames):
                fname = filenames[idx]
                if fname not in seen:
                    seen.add(fname)
                    results.append((fname, texts[idx]))
                if len(results) >= top_k:
                    break
        return results
    except Exception as e:
        print("❌ query_cv error:", e)
        traceback.print_exc()
        return []

# Uses GPT (via OpenRouter) to generate a ranked, grounded answer.

   # Steps:
   # - Combine retrieved CV chunks into a single context string
   # - Truncate context to avoid exceeding token limits
   # - Build a clear system/user prompt instructing the LLM to:
   #     Only use provided context (no hallucinations)
   #     Rank candidates if multiple matches
   #     Be concise (1–2 lines per candidate)
   # - Return the model's final text response
def generate_answer(question: str, retrieved: list[tuple[str, str]]) -> str:
    try:
        if not retrieved:
            return "No matching candidates found in the indexed CVs."
        context = "\n\n".join([f"From {fname}:\n{text}" for fname, text in retrieved])
        context = safe_truncate(context)

        prompt = f"""
You are a senior technical recruiter. Analyze the following candidate CV excerpts and answer the user's question precisely and concisely.

Question: {question}

CV Context:
{context}

Instructions:
- Base your answer ONLY on the CV Context.
- If multiple candidates match, rank them and explain why in 1–2 short lines each.
- If nothing matches, say so clearly.
- Prefer specific skills, companies, dates, projects, and metrics when available.
"""

        resp = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You help screen CVs for technical roles with clear, grounded answers."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=500,
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print("❌ generate_answer error:", e)
        traceback.print_exc()
        return "⚠️ Sorry, I couldn't generate an answer. Check API key/model or internet connection."

# Routes used
@app.route("/", methods=["GET", "POST"])
def home():
    try:
        answer = ""
        if request.method == "POST":
            user_input = (request.form.get("question") or "").strip()
            if not user_input:
                answer = "Please type a question."
            else:
                retrieved = query_cv(user_input, top_k=8)
                answer = generate_answer(user_input, retrieved)
        return render_template("index.html", answer=answer)
    except Exception as e:
        print("❌ Route error:", e)
        traceback.print_exc()
        return render_template("index.html", answer="⚠️ Unexpected server error. See console for details.")

# Run app here boyyyyyy!
if __name__ == "__main__":
    app.run(debug=True, threaded=True)
