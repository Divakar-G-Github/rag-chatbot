import streamlit as st
import requests

API_URL = "https://rag-chatbot-bo3r.onrender.com"

st.set_page_config(
    page_title="RAG Document Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 RAG Document Chatbot")
st.markdown("Upload a PDF and ask questions about it!")

# ---- Sidebar — PDF Upload ----
with st.sidebar:
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

    if uploaded_file:
        if st.button("Process PDF", type="primary"):
            with st.spinner("Processing your document..."):
                response = requests.post(
                    f"{API_URL}/upload",
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file,
                            "application/pdf"
                        )
                    }
                )
                if response.status_code == 200:
                    st.success(f"✅ {response.json()['message']}")
                    st.session_state.pdf_processed = True
                else:
                    st.error("❌ Failed to process PDF")

    st.divider()
    st.markdown("**Powered by**")
    st.markdown("🦙 Llama 3 via Groq")
    st.markdown("🔗 LangChain")
    st.markdown("🗄️ ChromaDB")

# ---- Main — Chat Area ----
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
question = st.chat_input("Ask something about your document...")

if question:
    # Show user message
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })
    with st.chat_message("user"):
        st.write(question)

    # Get answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = requests.post(
                f"{API_URL}/ask",
                json={"question": question}
            )
            if response.status_code == 200:
                data = response.json()
                answer = data["answer"]
                sources = data["sources"]

                st.write(answer)

                with st.expander("📚 View Source Chunks"):
                    for i, source in enumerate(sources):
                        st.markdown(f"**Chunk {i+1}:**")
                        st.markdown(f"{source[:300]}...")
                        st.divider()

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })
            else:
                st.error("❌ Could not get answer. Make sure PDF is uploaded first.")