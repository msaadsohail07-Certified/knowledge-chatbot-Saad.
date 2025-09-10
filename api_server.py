from fastapi import FastAPI
import wikipedia

# App create
app = FastAPI()

# Wikipedia English set
wikipedia.set_lang("en")

@app.get("/ask")
def ask(question: str):
    try:
        results = wikipedia.search(question, results=3)
        if not results:
            return {"error": "No Wikipedia page found"}
        
        answers = []
        for title in results:
            try:
                page = wikipedia.page(title, auto_suggest=False)
                answers.append({
                    "title": title,
                    "summary": page.summary[:500],
                    "url": page.url
                })
            except:
                continue
        return {"question": question, "answers": answers}
    except Exception as e:
        return {"error": str(e)}
