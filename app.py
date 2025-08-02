import os
import streamlit as st
from dotenv import load_dotenv

from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.pinecone import Pinecone as LangchainPinecone

from pinecone import Pinecone as PineconeClient, ServerlessSpec

# Load secrets (Streamlit or .env)
load_dotenv()
openai_api_key = st.secrets["OPENAI_API_KEY"]
pinecone_api_key = st.secrets["PINECONE_API_KEY"]

# UI
st.set_page_config(page_title="AICA RAG Chatbot", page_icon="🧠")
st.title("AICA RAG Chatbot 🤖")

# Initialize Pinecone v3 client
pc = PineconeClient(api_key=pinecone_api_key)

index_name = "aica-chatbot"

# Create index if it doesn't exist (optional safeguard)
if index_name not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Connect to existing index (this returns the `Index` object required by LangChain)
index = pc.Index(index_name)

# Embedding model
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Vectorstore from LangChain (must pass the Pinecone Index object)
vectorstore = LangchainPinecone(
    index=index,
    embedding=embeddings,
    text_key="text"
)

# QA pipeline
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(openai_api_key=openai_api_key),
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# User prompt
query = st.text_input("Ask your question:")

if query:
    with st.spinner("Thinking..."):
        result = qa.run(query)
        st.success(result)
