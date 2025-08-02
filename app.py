import streamlit as st
import os
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Streamlit secrets for keys
openai_api_key = st.secrets["OPENAI_API_KEY"]
pinecone_api_key = st.secrets["PINECONE_API_KEY"]
pinecone_env = st.secrets["PINECONE_ENV"]

# Initialize Pinecone
pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)

# Set up embeddings
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Load existing index from Pinecone
index = Pinecone.from_existing_index(index_name="aica-chatbot", embedding=embeddings)

# Streamlit UI
st.set_page_config(page_title="AICA RAG Chatbot")
st.title("🤖 AICA RAG Chatbot")
st.markdown("Upload documents (PDF, DOCX, XLSX) and ask questions in natural language.")

# Query input
query = st.text_input("Ask a question based on your uploaded docs or the indexed database:")

if query:
    llm = OpenAI(openai_api_key=openai_api_key, temperature=0)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=index.as_retriever())
    result = qa.run(query)
    st.write("### 📌 Answer:")
    st.success(result)
