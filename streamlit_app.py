import streamlit as st
import faiss, pickle, os
from sentence_transformers import SentenceTransformer
import openai  # ✅ added for direct OpenAI API

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
    ["Direct (local engine)", "GPT (OpenAI)"],  # ✅ changed Server mode to GPT
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

# ---------------------------
# Main Page
# ---------------------------
st.title("🌐 Wikipedia Chatbot — By Saad Sohail")
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

        elif mode.startswith("GPT"):
            # ✅ Direct OpenAI API call
            try:
                openai.api_key = st.secrets["OPENAI_API_KEY"]
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": query}]
                )
                answer = response['choices'][0]['message']['content']
                st.subheader("🤖 GPT Answer")
                st.write(answer)
            except Exception as e:
                st.error(f"❌ Error calling OpenAI API: {e}")

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
    - GPT mode uses OpenAI API. Make sure `OPENAI_API_KEY` is set in Streamlit secrets.  
    """
)
