import os
import streamlit as st
from dotenv import load_dotenv

from langchain.chains import RetrievalQA
from langchain_community.vectorstores.pinecone import Pinecone as LangchainPinecone
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

# Load secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]
pinecone_api_key = st.secrets["PINECONE_API_KEY"]
pinecone_env = st.secrets["PINECONE_ENVIRONMENT"]

# Streamlit UI
st.set_page_config(page_title="AICA RAG Chatbot", page_icon="🤖")
st.title("AICA RAG Chatbot 🤖")

# Initialize Pinecone v3 client
pc = Pinecone(api_key=pinecone_api_key)

index_name = "aica-chatbot"

# Create index if it doesn't exist
if index_name not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=pinecone_env)
    )

# Connect to existing index
index = pc.Index(index_name)

# Embedding & Vector Store
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
vectorstore = LangchainPinecone(index=index, embedding=embeddings, text_key="text")

# QA chain
llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-4")
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

# User input
query = st.text_input("Ask your question 👇")

if query:
    with st.spinner("Thinking..."):
        response = qa_chain.run(query)
        st.markdown(f"**Answer:** {response}")
