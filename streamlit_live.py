import streamlit as st
import wikipedia

# Set language
wikipedia.set_lang("en")

# Streamlit UI
st.set_page_config(page_title="Live Wiki Chatbot", layout="wide")
st.title("🌐 Live Wiki Chatbot — By Saad Sohail")
st.caption("Ask anything! Answers are fetched live from Wikipedia.")

query = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if query.strip() == "":
        st.warning("⚠️ Please enter a question.")
    else:
        try:
            # Search Wikipedia
            results = wikipedia.search(query, results=3)
            if not results:
                st.error("No relevant Wikipedia page found.")
            else:
                st.subheader("📖 Top Wikipedia Results")
                for i, title in enumerate(results, 1):
                    try:
                        page = wikipedia.page(title, auto_suggest=False)
                        st.markdown(f"**Result {i}: {title}**")
                        st.write(page.summary[:700] + ("..." if len(page.summary) > 700 else ""))
                        st.markdown(f"[🔗 Read full article on Wikipedia]({page.url})")
                        st.markdown("---")
                    except Exception as e:
                        st.error(f"Error fetching page {title}: {e}")
        except Exception as e:
            st.error(f"❌ Error: {e}")
