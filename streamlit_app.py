import streamlit as st
import os
import pickle
import numpy as np
from google import genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="PDC Knowledge Chatbot", page_icon="🤖", layout="wide")

st.markdown("""
<style>
body { background-color: #0F0F1A; }
.stApp { background-color: #0F0F1A; color: #E0E0FF; }
</style>
""", unsafe_allow_html=True)

# Load documents
@st.cache_resource
def load_documents():
    chunks = []
    data_dir = "data"
    if os.path.exists(data_dir):
        for fname in os.listdir(data_dir):
            if fname.endswith(".txt"):
                with open(os.path.join(data_dir, fname), "r", encoding="utf-8") as f:
                    text = f.read()
                    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 100]
                    chunks.extend(paragraphs)
    return chunks

@st.cache_resource
def build_vectorizer(chunks):
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(chunks)
    return vectorizer, matrix

chunks = load_documents()
vectorizer, matrix = build_vectorizer(chunks)

# Sidebar
st.sidebar.title("⚙️ Settings")
mode = st.sidebar.radio("Operation mode", ["Direct (local engine)", "Gemini (Google AI)"])
n_results = st.sidebar.slider("Number of results to retrieve (1-5)", 1, 5, 2)
max_chars = st.sidebar.slider("Max characters to display per result (100-1500)", 100, 1500, 500)

# Main
st.title("🌐 PDC Knowledge Chatbot — By Saad")
st.markdown(f"**Mode:** {mode}")

query = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if query.strip() == "":
        st.warning("⚠️ Please enter a question.")
    else:
        if mode.startswith("Direct"):
            if len(chunks) == 0:
                st.error("❌ No data found. Run fetch_wiki.py first!")
            else:
                query_vec = vectorizer.transform([query])
                scores = cosine_similarity(query_vec, matrix)[0]
                top_idx = np.argsort(scores)[::-1][:n_results]
                
                st.subheader("⬛ Local Engine Results")
                for rank, idx in enumerate(top_idx, 1):
                    st.markdown(f"**Result {rank}:**")
                    st.write(chunks[idx][:max_chars] + ("..." if len(chunks[idx]) > max_chars else ""))
                    st.markdown("---")

        elif mode.startswith("Gemini"):
            try:
                api_key = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
                if not api_key:
                    st.error("❌ Gemini API key not found. Please set GEMINI_API_KEY in secrets.")
                else:
                    client = genai.Client(api_key=api_key)
                    
                    # RAG - retrieve context first
                    query_vec = vectorizer.transform([query])
                    scores = cosine_similarity(query_vec, matrix)[0]
                    top_idx = np.argsort(scores)[::-1][:2]
                    context = "\n\n".join([chunks[i] for i in top_idx if scores[i] > 0])
                    
                    prompt = f"""You are a PDC (Parallel and Distributed Computing) expert assistant.
                    
Context from knowledge base:
{context}

Question: {query}

Answer based on the context above. If context is not relevant, answer from your knowledge."""
                    
                    resp = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
                    st.subheader("🤖 Gemini Answer")
                    st.markdown(resp.text)
            except Exception as e:
                st.error(f"❌ Error: {e}")

st.markdown("""
<div style='margin-top:40px;padding:16px;background:#1A1A2E;border-radius:10px;font-size:13px;color:#A0A0CC'>
<b>Notes</b><br>
- Local engine uses TF-IDF retrieval from PDC knowledge base<br>
- Gemini mode uses Google AI with RAG context<br>
- Make sure GEMINI_API_KEY is set in .streamlit/secrets.toml
</div>
""", unsafe_allow_html=True)