import wikipedia
import os

TOPICS = [
    "Parallel computing",
    "Distributed computing",
    "Message Passing Interface",
    "OpenMP",
    "CUDA parallel computing",
    "MapReduce",
    "Flynn's taxonomy",
    "Amdahl's law",
    "Thread computing",
    "Process synchronization",
    "Deadlock",
    "Load balancing computing",
    "Apache Hadoop",
    "Apache Spark",
    "Cloud computing"
]

os.makedirs("data", exist_ok=True)

for topic in TOPICS:
    try:
        page = wikipedia.page(topic)
        filename = f"data/{topic.replace(' ', '_')}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(page.content)
        print(f"✅ Saved: {topic}")
    except Exception as e:
        print(f"❌ Failed: {topic} — {e}")