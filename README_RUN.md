# RAG Chatbot — CCP Project

## How to Run

1. Create a virtual environment:
   python -m venv venv
   venv\Scripts\activate    # Windows
   source venv/bin/activate # Mac/Linux

2. Install dependencies:
   pip install -r requirements.txt

3. Put your documents (PDF or TXT files) in the `data/` folder.

4. Build the index (this will create the `index/` folder):
   python build_index.py --data_dir data --index_dir index

5. (Optional) If you have an OpenAI API key, set it:
   - Windows PowerShell:
     setx OPENAI_API_KEY "your_api_key_here"
   - Mac/Linux:
     export OPENAI_API_KEY="your_api_key_here"

6. Run the Streamlit app:
   streamlit run streamlit_app.py

7. Open the local URL shown in the terminal (e.g., http://localhost:8501) and start chatting with the bot.

---

## Notes
- All your documents should be placed in the `data/` folder.
- The `index/` folder will be generated automatically.
- If you don't set an OpenAI API key, you can still use the **Local (transformers)** option in the Streamlit app (quality may be lower).
