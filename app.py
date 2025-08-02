import os
import streamlit as st
from dotenv import load_dotenv

from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone

import pinecone  # Legacy client v2.x

# Load API Keys
load_dotenv()
openai_api_key = st.secrets["OPENAI_API_KEY"]
pinecone_api_key = st.secrets["PINECONE_API_KEY"]

# Streamlit UI
st.set_page_config(page_title="AICA RAG Chatbot", page_icon="🧠")
st.title("AICA RAG Chatbot 🤖")

# Initialize legacy Pinecone client (v2.x)
pinecone.init(api_key=pinecone_api_key, environment="us-east-1")  # Set correct region

index_name = "aica-chatbot"

# Ensure index exists
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=1536, metric="cosine")

# LangChain-compatible Pinecone connection
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

vectorstore = LangchainPinecone.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
    text_key="text"
)

# QA Chain
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(openai_api_key=openai_api_key),
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# Input and Response
query = st.text_input("Ask your question:")

if query:
    with st.spinner("Thinking..."):
        result = qa.run(query)
        st.success(result)
