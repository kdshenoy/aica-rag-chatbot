import os
import streamlit as st
from dotenv import load_dotenv

from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone

from pinecone import Pinecone

# Load environment variables
load_dotenv()

# Load secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]
pinecone_api_key = st.secrets["PINECONE_API_KEY"]

# Streamlit UI setup
st.set_page_config(page_title="AICA RAG Chatbot", page_icon="🧠")
st.title("AICA RAG Chatbot 🤖")

# Initialize Pinecone client
pc = Pinecone(api_key=pinecone_api_key)
index_name = "aica-chatbot"
index = pc.Index(index_name)

# Set up embeddings
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Use Langchain Pinecone wrapper
vectorstore = LangchainPinecone.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# QA Chain setup
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(openai_api_key=openai_api_key),
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# User Query Input
query = st.text_input("Ask your question related to the PDF content")

if query:
    with st.spinner("Generating Answer..."):
        response = qa.run(query)
        st.success(response)
