import os
import streamlit as st
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

load_dotenv()

st.set_page_config(page_title="AICA RAG Chatbot", page_icon="🤖")
st.title("🤖 AICA RAG Chatbot")

# Setup Pinecone Index
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
index = Pinecone.from_existing_index(index_name="aica-chatbot", embedding=embeddings)

uploaded_file = st.file_uploader("📄 Upload a file (PDF/DOCX/XLSX)", type=["pdf", "docx", "xlsx"])
if uploaded_file and st.button("📥 Ingest & Index"):
    with open("uploaded_file.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    from ingest import ingest_documents
    ingest_documents("uploaded_file.pdf", "aica-chatbot")
    st.success("✅ Document ingested and indexed!")

query = st.text_input("💬 Ask a question")
if query and st.button("🔍 Get Answer"):
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(model_name="gpt-4", openai_api_key=os.getenv("OPENAI_API_KEY")),
        chain_type="stuff",
        retriever=index.as_retriever()
    )
    answer = qa.run(query)
    st.write("🧠 Answer:", answer)
