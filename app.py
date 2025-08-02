import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# UI title
st.title("AICA RAG Chatbot 🤖")

# Initialize Pinecone v3 client
pc = Pinecone(api_key=pinecone_api_key)

# Index setup
index_name = "aica-chatbot"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",  # or "dotproduct" or "euclidean"
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

# Connect to the index
index = pc.Index(index_name)

# Create embedding model
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Set up vector store using LangChain wrapper
vectorstore = LangchainPinecone(
    index=index,
    embedding=embeddings,
    text_key="text"
)

# Basic user input for search
query = st.text_input("Ask your question:")

if query:
    results = vectorstore.similarity_search(query, k=3)
    st.write("Top Results:")
    for i, doc in enumerate(results):
        st.write(f"**{i+1}.** {doc.page_content}")
