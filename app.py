import os
import streamlit as st
from dotenv import load_dotenv

from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.pinecone import Pinecone as LangchainPinecone

from pinecone import Pinecone as PineconeClient, ServerlessSpec

# Load secrets
load_dotenv()
openai_api_key = st.secrets["OPENAI_API_KEY"]
pinecone_api_key = st.secrets["PINECONE_API_KEY"]

# UI setup
st.set_page_config(page_title="AICA RAG Chatbot", page_icon="🧠")
st.title("AICA RAG Chatbot 🤖")

# Initialize Pinecone client (v3)
pc = PineconeClient(api_key=pinecone_api_key)

index_name = "aica-chatbot"

# Create index if it doesn't exist
if index_name not in [index.name for index in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Set up OpenAI embeddings
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Load vectorstore from existing Pinecone index
vectorstore = LangchainPinecone.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Build QA chain
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(openai_api_key=openai_api_key),
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# User input box
query = st.text_input("Ask your question:")

if query:
    with st.spinner("Thinking..."):
        result = qa.run(query)
        st.success(result)
