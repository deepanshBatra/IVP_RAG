# app.py
import streamlit as st
from modules.llm import generate_response as chat_with_groq
from modules.rag_pipeline import build_vector_db, retrieve_docs

# -----------------------------
# Load Vector DB from company docs
# -----------------------------
db = build_vector_db("data/IVP Global Travel Policy V-3.0.pdf")

# -----------------------------
# Streamlit Chat UI
# -----------------------------
st.set_page_config(layout="wide")
st.set_page_config(page_title="Internal Policies Chatbot", page_icon="🤖")
st.title("📚 IVP Internal Policies Chatbot")
st.info(
    "ℹ️ This chatbot is designed to answer questions based on the **IVP Global Travel Policy**" \
)


user_q = st.text_input("Ask me anything about the company:")

if user_q:
    # 1. Retrieve relevant docs
    docs = retrieve_docs(user_q, db)

    # 2. Build context for Groq
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"""
    You are an AI assistant helping employees understand the IVP Global Travel Policy.

    Your task is to answer questions based only on the given CONTEXT.

    Guidelines:
    1. If the user asks a factual question → give a short, direct answer (2-3 sentences).
    2. If the user asks for a summary or explanation → provide a clear, structured summary in bullet points.
    3. If multiple sections of the context are relevant → merge them into a coherent answer.
    4. If the context does not contain the answer → reply with:
    "I could not find this information in the IVP Global Travel Policy."
    5. Do not invent or add extra information outside the context.

    Context:
    {context}

    Question: {user_q}
    Answer:
    """

    # 3. Get response from Groq
    answer = chat_with_groq(prompt)

    # 4. Show answer
    st.write(answer)

    # (Optional) show retrieved context
    with st.expander("🔍 Retrieved Context"):
        st.write(context)
