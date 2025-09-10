import wikipedia
import os

# Wikipedia language set karna (default English)
wikipedia.set_lang("en")

# Topics jinke pages chahiye (exact titles)
topics = [
    "Machine learning",
    "Natural language processing",
    "Information retrieval"
]

# data folder banaye agar na ho
os.makedirs("data", exist_ok=True)

for t in topics:
    try:
        # Direct title se fetch karo
        page = wikipedia.page(title=t, auto_suggest=False)
        text = page.content

        safe_name = t.replace(" ", "_").replace("/", "_")
        with open(f"data/{safe_name}.txt", "w", encoding="utf-8") as f:
            f.write(text)
        print(f"✅ Saved {t}")
    except Exception as e:
        print(f"❌ Error fetching {t}: {e}")
