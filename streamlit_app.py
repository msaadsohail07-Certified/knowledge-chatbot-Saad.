import streamlit as st
import requests
import faiss, pickle, os
from sentence_transformers import SentenceTransformer

# ---------------------------
# Load local FAISS index (safe loader)
# ---------------------------
INDEX_DIR = "index"
chunks, meta, index = [], [], None

if os.path.exists(os.path.join(INDEX_DIR, "faiss.index")):
    index = faiss.read_index(os.path.join(INDEX_DIR, "faiss.index"))
    with open(os.path.join(INDEX_DIR, "chunks.pkl"), "rb") as f:
        data = pickle.load(f)

    # Robust loader: handle both dict and list
    if isinstance(data, dict):
        chunks = data.get("chunks", [])
        meta = data.get("meta", [])
    elif isinstance(data, list):
        if len(data) > 0 and isinstance(data[0], str):
            chunks = data
            meta = [{} for _ in chunks]
        elif len(data) > 0 and isinstance(data[0], dict):
            if "text" in data[0]:
                chunks = [d.get("text", "") for d in data]
                meta = [d.get("meta", {}) for d in data]
            elif "chunk" in data[0]:
                chunks = [d.get("chunk", "") for d in data]
                meta = [d.get("meta", {}) for d in data]
            else:
                chunks = [str(d) for d in data]
                meta = [{} for _ in chunks]
        else:
            chunks = [str(d) for d in data]
            meta = [{} for _ in chunks]

# ---------------------------
# Model load
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

model = load_model()

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="Wikipedia Chatbot", layout="wide")

# ---------------------------
# Sidebar (Settings)
# ---------------------------
st.sidebar.header("⚙️ Settings")

mode = st.sidebar.radio(
    "Operation mode",
    ["Direct (local engine)", "Server (use FastAPI)"],
    index=0
)

# Adjusted ranges
n_pages = st.sidebar.slider(
    "📄 Number of Wikipedia pages (1–4)",
    min_value=1,
    max_value=4,
    value=2
)

max_chars = st.sidebar.slider(
    "🔤 Max characters per page (100–1500)",
    min_value=100,
    max_value=1500,
    value=500
)

# Server URL only if Server mode selected
server_url = None
if mode.startswith("Server"):
    server_url = st.sidebar.text_input(
        "🌐 Server URL (e.g. http://127.0.0.1:8000)",
        "http://127.0.0.1:8000"
    )

# ---------------------------
# Main Page
# ---------------------------
st.title("🌐 Wikipedia Chatbot — By Saad Sohail")

# NEW: Show current mode under title
st.markdown(f"**Mode:** {mode}")

query = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if query.strip() == "":
        st.warning("⚠️ Please enter a question.")
    else:
        if mode.startswith("Direct") and index is not None:
            # Local FAISS search
            q_emb = model.encode([query], convert_to_numpy=True)
            D, I = index.search(q_emb.astype("float32"), k=n_pages)

            st.subheader("📖 Local Engine Results")
            for rank, idx in enumerate(I[0], 1):
                st.markdown(f"**Result {rank}:**")
                st.write(chunks[idx][:max_chars] + ("..." if len(chunks[idx]) > max_chars else ""))
                st.markdown("---")

        elif mode.startswith("Server"):
            # Server API call
            try:
                resp = requests.get(f"{server_url}/ask", params={"question": query})
                data = resp.json()
                if "answers" in data:
                    st.subheader("📖 Wikipedia Results")
                    for i, ans in enumerate(data["answers"], 1):
                        st.markdown(f"**Result {i}: {ans['title']}**")
                        st.write(ans["summary"])
                        st.markdown(f"[🔗 Read full article]({ans['url']})")
                        st.markdown("---")
                else:
                    st.error("No results from API.")
            except Exception as e:
                st.error(f"❌ Error calling API: {e}")
        else:
            st.error("⚠️ Local Engine not available. Build the index first.")

# ---------------------------
# Notes Section
# ---------------------------
st.info(
    """
    **Notes**
    - The first run may download model weights (internet required).  
    - The answers are extractive: taken from retrieved Wikipedia summaries.  
    - If you use Server mode, make sure the FastAPI server is running (`uvicorn api_server:app --port 8000`).  
    """
)
