import os
import streamlit as st
from dotenv import load_dotenv

from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.pinecone import Pinecone as LangchainPinecone

from pinecone import Pinecone, ServerlessSpec

# Load .env or Streamlit secrets
load_dotenv()
openai_api_key = st.secrets["OPENAI_API_KEY"]
pinecone_api_key = st.secrets["PINECONE_API_KEY"]

# UI Setup
st.set_page_config(page_title="AICA RAG Chatbot", page_icon="🧠")
st.title("AICA RAG Chatbot 🤖")

# Initialize Pinecone Client (v3)
pc = Pinecone(api_key=pinecone_api_key)

index_name = "aica-chatbot"

# Create index if not exists (optional)
if index_name not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

# Set up embeddings
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Use LangChain Pinecone wrapper
vectorstore = LangchainPinecone(
    index=index,
    embedding=embeddings,
    text_key="text"
)

# QA Chain setup
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(openai_api_key=openai_api_key),
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# User input
query = st.text_input("Ask your question")

if query:
    with st.spinner("Thinking..."):
        response = qa.run(query)
        st.success(response)
