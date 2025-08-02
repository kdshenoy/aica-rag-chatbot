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

# Streamlit secrets for keys
openai_api_key = st.secrets["OPENAI_API_KEY"]
pinecone_api_key = st.secrets["PINECONE_API_KEY"]

# Page Configuration
st.set_page_config(page_title="AICA RAG Chatbot", page_icon="🧠")
st.title("AICA RAG Chatbot 🤖")

# Initialize Pinecone (new method)
pc = Pinecone(api_key=pinecone_api_key)

# Load existing index
index_name = "aica-chatbot"
index = pc.Index(index_name)

# Set up OpenAI Embeddings
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Load vector store using LangChain’s Pinecone wrapper
vectorstore = LangchainPinecone(index, embeddings.embed_query, "text")

# Set up RetrievalQA chain
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(openai_api_key=openai_api_key),
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# Streamlit UI
query = st.text_input("Ask your question related to the PDF content")

if query:
    with st.spinner("Generating Answer..."):
        response = qa.run(query)
        st.success(response)
