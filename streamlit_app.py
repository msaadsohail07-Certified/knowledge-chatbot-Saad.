import streamlit as st
import os
import numpy as np
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="PDC RAG Knowledge Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Theme State ────────────────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

def toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"

is_dark = st.session_state.theme == "dark"

# ─── Theme Colors ───────────────────────────────────────────────────
if is_dark:
    bg_main        = "#0A0A14"
    bg_card        = "#0F0F1E"
    bg_input       = "#0F0F1E"
    border_color   = "#2a2a45"
    border_subtle  = "#1e1e35"
    text_primary   = "#e0e0ff"
    text_secondary = "#a0a0cc"
    text_muted     = "#5a5a8a"
    accent         = "#7c7cff"
    accent_hover   = "#9090ff"
    accent_soft    = "#1a1a30"
    accent_border  = "#3a3a60"
    sidebar_bg     = "#0F0F1E"
    chip_bg        = "#0a0a18"
    result_border  = "#7c7cff"
    badge_bg       = "#0d200d"
    badge_color    = "#5fd45f"
    badge_border   = "#1e3e1e"
    toggle_icon    = "☀️"
    toggle_label   = "Light Mode"
    footer_bg      = "#0F0F1E"
else:
    bg_main        = "#f5f5fa"
    bg_card        = "#ffffff"
    bg_input       = "#ffffff"
    border_color   = "#ddddf0"
    border_subtle  = "#e8e8f5"
    text_primary   = "#1a1a3a"
    text_secondary = "#4a4a7a"
    text_muted     = "#9090b8"
    accent         = "#5a5aff"
    accent_hover   = "#4040ee"
    accent_soft    = "#eeeeff"
    accent_border  = "#c0c0ff"
    sidebar_bg     = "#ffffff"
    chip_bg        = "#f0f0ff"
    result_border  = "#5a5aff"
    badge_bg       = "#e8f5e8"
    badge_color    = "#2a7a2a"
    badge_border   = "#b0d8b0"
    toggle_icon    = "🌙"
    toggle_label   = "Dark Mode"
    footer_bg      = "#ffffff"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

.stApp {{
    background-color: {bg_main};
    color: {text_primary};
}}

section[data-testid="stSidebar"] {{
    background-color: {sidebar_bg} !important;
    border-right: 1px solid {border_subtle};
}}

section[data-testid="stSidebar"] * {{
    color: {text_secondary} !important;
}}

section[data-testid="stSidebar"] label p {{
    color: {text_primary} !important;
    font-size: 13px !important;
}}

section[data-testid="stSidebar"] .stSelectbox > div > div {{
    background: {accent_soft} !important;
    border-color: {accent_border} !important;
    color: {text_primary} !important;
}}

/* Header */
.chat-header {{
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 20px 0 8px 0;
    border-bottom: 1px solid {border_subtle};
    margin-bottom: 24px;
}}
.chat-title {{
    font-size: 20px;
    font-weight: 600;
    color: {text_primary};
}}
.chat-subtitle {{
    font-size: 12px;
    color: {text_muted};
    margin-top: 2px;
}}
.online-badge {{
    font-size: 11px;
    padding: 4px 12px;
    border-radius: 20px;
    background: {badge_bg};
    color: {badge_color};
    border: 1px solid {badge_border};
}}
.toggle-btn-wrap {{
    margin-left: auto;
    display: flex;
    align-items: center;
    gap: 10px;
}}

/* Input */
.stTextInput > div > div > input {{
    background: {bg_input} !important;
    border: 1px solid {border_color} !important;
    border-radius: 12px !important;
    color: {text_primary} !important;
    padding: 14px 16px !important;
    font-size: 14px !important;
}}
.stTextInput > div > div > input:focus {{
    border-color: {accent} !important;
    box-shadow: 0 0 0 2px {accent}26 !important;
}}
.stTextInput > div > div > input::placeholder {{
    color: {text_muted} !important;
}}

/* Buttons */
.stButton > button {{
    background: {accent} !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 28px !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    transition: all 0.2s ease !important;
}}
.stButton > button:hover {{
    background: {accent_hover} !important;
    transform: translateY(-1px);
}}

/* Answer card */
.answer-card {{
    background: {bg_card};
    border: 1px solid {border_color};
    border-radius: 14px;
    padding: 20px;
    margin-top: 16px;
    animation: fadeIn 0.4s ease;
}}
@keyframes fadeIn {{
    from {{ opacity: 0; transform: translateY(8px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}
.answer-header {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 14px;
    padding-bottom: 12px;
    border-bottom: 1px solid {border_subtle};
}}
.answer-model-badge {{
    margin-left: auto;
    font-size: 11px;
    padding: 3px 10px;
    border-radius: 20px;
    background: {accent_soft};
    color: {accent};
    border: 1px solid {accent_border};
}}
.answer-body {{
    font-size: 14px;
    color: {text_secondary};
    line-height: 1.75;
}}
.answer-title {{
    font-size: 14px;
    font-weight: 500;
    color: {text_primary};
}}

/* Source pill */
.source-pill {{
    display: flex;
    align-items: center;
    gap: 8px;
    background: {bg_main};
    border: 1px solid {border_subtle};
    border-radius: 8px;
    padding: 8px 14px;
    margin-top: 14px;
    font-size: 12px;
    color: {text_muted};
}}

/* Result card */
.result-card {{
    background: {bg_card};
    border: 1px solid {border_color};
    border-left: 3px solid {result_border};
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 12px;
    animation: fadeIn 0.4s ease;
}}
.result-rank {{
    font-size: 11px;
    color: {accent};
    font-weight: 500;
    margin-bottom: 6px;
}}
.result-text {{
    font-size: 13px;
    color: {text_secondary};
    line-height: 1.7;
}}

/* Chips */
.chips-row {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 14px;
    margin-bottom: 8px;
}}
.chip {{
    font-size: 12px;
    padding: 5px 12px;
    border-radius: 20px;
    border: 1px solid {border_color};
    color: {text_muted};
    background: {chip_bg};
}}

/* Sidebar label */
.sidebar-section-label {{
    font-size: 11px;
    font-weight: 500;
    color: {text_muted};
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 10px;
    margin-top: 20px;
}}

/* Footer */
.footer-note {{
    margin-top: 40px;
    padding: 14px 18px;
    background: {footer_bg};
    border-radius: 10px;
    border: 1px solid {border_subtle};
    font-size: 12px;
    color: {text_muted};
    line-height: 1.8;
}}
.footer-note b {{
    color: {text_secondary};
}}
.rag-badge {{
    font-size: 11px;
    padding: 4px 12px;
    border-radius: 20px;
    background: #1a1030;
    color: #a070ff;
    border: 1px solid #4a2a80;
    margin-left: 8px;
}}
</style>
""", unsafe_allow_html=True)


# ─── Data Loading ────────────────────────────────────────────────────
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


# ─── Sidebar ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f'<div class="sidebar-section-label">⚙️ Configuration</div>', unsafe_allow_html=True)

    mode = st.radio(
        "Operation Mode",
        ["Groq AI (Fast)", "Local Engine (TF-IDF)"],
        index=0
    )

    st.markdown(f'<div class="sidebar-section-label">🔧 Retrieval Settings</div>', unsafe_allow_html=True)
    n_results = st.slider("Results to retrieve", 1, 5, 2)
    max_chars = st.slider("Max characters per result", 100, 1500, 500)

    if mode.startswith("Groq"):
        st.markdown(f'<div class="sidebar-section-label">🤖 Model</div>', unsafe_allow_html=True)
        model_choice = st.selectbox(
            "Groq Model",
            ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "gemma2-9b-it"],
            index=0
        )
    else:
        model_choice = "llama3-8b-8192"

    st.markdown("---")
    st.markdown(
        f'<div style="font-size:11px; color:{text_muted}; text-align:center;">PDC Chatbot · By Saad<br>RAG + Groq AI</div>',
        unsafe_allow_html=True
    )


# ─── Header ──────────────────────────────────────────────────────────
col_title, col_toggle = st.columns([6, 1])

with col_title:
    st.markdown(f"""
    <div class="chat-header">
        <div style="font-size:26px">🧠</div>
        <div>
            <div class="chat-title">PDC RAG Knowledge Chatbot</div>
            <div class="chat-subtitle">Parallel &amp; Distributed Computing · CCP Project · By Saad</div>
        </div>
        <div class="online-badge">● Online</div>
        <div class="rag-badge">⚡ RAG-Powered</div>
    </div>
    """, unsafe_allow_html=True)

with col_toggle:
    st.markdown("<div style='margin-top:18px'></div>", unsafe_allow_html=True)
    if st.button(f"{toggle_icon} {toggle_label}", key="theme_toggle"):
        toggle_theme()
        st.rerun()


# ─── Input ───────────────────────────────────────────────────────────
col1, col2 = st.columns([5, 1])
with col1:
    query = st.text_input(
        "",
        placeholder="Ask something about PDC... e.g. What is MPI?",
        label_visibility="collapsed"
    )
with col2:
    ask_btn = st.button("Ask ↗", use_container_width=True)

st.markdown(f"""
<div class="chips-row">
    <div class="chip">What is parallel computing?</div>
    <div class="chip">Explain MPI</div>
    <div class="chip">OpenMP vs CUDA</div>
    <div class="chip">What is MapReduce?</div>
    <div class="chip">Types of parallelism</div>
</div>
""", unsafe_allow_html=True)


# ─── Logic ───────────────────────────────────────────────────────────
if ask_btn:
    if query.strip() == "":
        st.warning("⚠️ Please enter a question first.")

    elif mode.startswith("Local"):
        if len(chunks) == 0:
            st.error("❌ No data found. Run fetch_wiki.py first.")
        else:
            query_vec = vectorizer.transform([query])
            scores = cosine_similarity(query_vec, matrix)[0]
            top_idx = np.argsort(scores)[::-1][:n_results]

            st.markdown(f"""
            <div style="margin-top:20px; font-size:13px; color:{text_muted}; margin-bottom:12px;">
                📚 Showing top {n_results} results from knowledge base
            </div>
            """, unsafe_allow_html=True)

            for rank, idx in enumerate(top_idx, 1):
                text = chunks[idx][:max_chars] + ("..." if len(chunks[idx]) > max_chars else "")
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-rank">Result {rank}</div>
                    <div class="result-text">{text}</div>
                </div>
                """, unsafe_allow_html=True)

    elif mode.startswith("Groq"):
        api_key = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", None)
        if not api_key:
            st.error("❌ GROQ_API_KEY not found. Add it in .streamlit/secrets.toml")
        else:
            with st.spinner("Thinking..."):
                try:
                    query_vec = vectorizer.transform([query])
                    scores = cosine_similarity(query_vec, matrix)[0]
                    top_idx = np.argsort(scores)[::-1][:2]
                    context = "\n\n".join([chunks[i] for i in top_idx if scores[i] > 0])
                    chunk_count = len([i for i in top_idx if scores[i] > 0])

                    prompt = f"""You are a PDC (Parallel and Distributed Computing) expert assistant.

Context from knowledge base:
{context}

Question: {query}

Answer clearly and concisely. If context is not relevant, answer from your knowledge."""

                    client = Groq(api_key=api_key)
                    response = client.chat.completions.create(
                        model=model_choice,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.7,
                        max_tokens=1024
                    )
                    answer = response.choices[0].message.content

                    st.markdown(f"""
                    <div class="answer-card">
                        <div class="answer-header">
                            <span style="font-size:18px">⚡</span>
                            <span class="answer-title">Answer</span>
                            <div class="answer-model-badge">{model_choice}</div>
                        </div>
                        <div class="answer-body">{answer}</div>
                        <div class="source-pill">
                            📦 Retrieved from knowledge base · {chunk_count} context chunks used
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"❌ Error: {e}")


# ─── Footer ──────────────────────────────────────────────────────────
st.markdown(f"""
<div class="footer-note">
    <b>Notes</b><br>
    · Local engine uses TF-IDF retrieval from PDC knowledge base<br>
    · Groq AI mode uses RAG context + LLaMA / Mixtral models (free tier)<br>
    · Add <code>GROQ_API_KEY</code> in <code>.streamlit/secrets.toml</code>
</div>
""", unsafe_allow_html=True)
