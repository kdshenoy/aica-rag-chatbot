import os
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone

# Load Streamlit secrets
openai_key = st.secrets["OPENAI_API_KEY"]
pinecone_key = st.secrets["PINECONE_API_KEY"]
pinecone_env = st.secrets["PINECONE_ENVIRONMENT"]

# Load environment variables (if .env is used locally)
load_dotenv()

# ─────────────────────────────────────────────────────────────────────
# Page Configuration
st.set_page_config(page_title="AICA RAG Chatbot", page_icon="🤖")

st.title("🤖 AICA RAG Chatbot")
st.markdown("Upload documents (PDF, DOCX, XLSX) and ask questions in natural language.")

# ─────────────────────────────────────────────────────────────────────
# Initialize Pinecone vector index
embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
index = Pinecone.from_existing_index("aica-chatbot", embeddings)

# ─────────────────────────────────────────────────────────────────────
# Upload Section
uploaded_file = st.file_uploader("📄 Upload your policy doc(s)", type=["pdf", "docx", "xlsx"])

if uploaded_file and st.button("📥 Ingest & Index"):
    with open("uploaded_doc", "wb") as f:
        f.write(uploaded_file.getbuffer())
    from ingest import ingest_documents
    ingest_documents("uploaded_doc", "aica-chatbot")
    st.success("✅ Document ingested successfully.")

# ─────────────────────────────────────────────────────────────────────
# Q&A Section
user_question = st.text_input("💬 Ask your question:")

if user_question and st.button("🎯 Get Answer"):
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(model="gpt-4", openai_api_key=openai_key),
        chain_type="stuff",
        retriever=index.as_retriever()
    )
    result = qa.run(user_question)
    st.markdown(f"🧠 **Answer:** {result}")
