import os
import streamlit as st
from dotenv import load_dotenv

from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.pinecone import Pinecone as LangchainPinecone

import pinecone  # v2 compatible

# Load secrets
load_dotenv()
openai_api_key = st.secrets["OPENAI_API_KEY"]
pinecone_api_key = st.secrets["PINECONE_API_KEY"]

# UI
st.set_page_config(page_title="AICA RAG Chatbot", page_icon="🧠")
st.title("AICA RAG Chatbot 🤖")

# Init Pinecone v2
pinecone.init(api_key=pinecone_api_key, environment="us-east-1")
index_name = "aica-chatbot"

# Create index if not exists
if index_name not in pinecone.list_indexes():
    pinecone.create_index(name=index_name, dimension=1536, metric="cosine")

# Connect to index
index = pinecone.Index(index_name)

# LangChain Vectorstore
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
vectorstore = LangchainPinecone(index=index, embedding=embeddings, text_key="text")

# Retrieval QA
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(openai_api_key=openai_api_key),
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# Streamlit input
query = st.text_input("Ask your question:")
if query:
    with st.spinner("Thinking..."):
        result = qa.run(query)
        st.success(result)
