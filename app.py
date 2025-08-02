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

# UI
st.set_page_config(page_title="AICA RAG Chatbot", page_icon="🧠")
st.title("AICA RAG Chatbot 🤖")

# Initialize Pinecone client
pc = PineconeClient(api_key=pinecone_api_key)

# Index name
index_name = "aica-chatbot"

# Create the index if it doesn't exist
if index_name not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Use Langchain’s Pinecone wrapper to load the vectorstore
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
vectorstore = LangchainPinecone.from_existing_index(index_name=index_name, embedding=embeddings)

# QA Chain
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(openai_api_key=openai_api_key),
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# UI for query
query = st.text_input("Ask your question:")

if query:
    with st.spinner("Thinking..."):
        result = qa.run(query)
        st.success(result)
