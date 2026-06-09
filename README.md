# PDC RAG Knowledge Chatbot — By Saad (CCP Project)

A RAG-based chatbot for Parallel and Distributed Computing (PDC) topics. 
Built using TF-IDF retrieval and Groq AI (LLaMA 3).

## Live Demo
👉 [Click here to try the chatbot](https://pdc-rag-chatbot-saad.streamlit.app/)

---

## Features
- PDC domain-specific knowledge base
- TF-IDF Retrieval + Groq AI (LLaMA 3) integration
- Modern dark/light theme Streamlit interface
- RAG-Powered (Retrieval Augmented Generation)
- Can run both locally and on the cloud

---

## Tech Stack
- Python, Streamlit
- Scikit-learn (TF-IDF)
- Groq API (LLaMA 3)

---

## How to Run Locally
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Secrets Setup
Add in `.streamlit/secrets.toml`:
```toml
GROQ_API_KEY = "your_key_here"
```
