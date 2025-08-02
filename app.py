import os
import streamlit as st
from dotenv import load_dotenv

from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone

from pinecone import Pinecone as PineconeClient, ServerlessSpec

# Load secrets
load_dotenv()
openai_api_key = st.secrets["OPENAI_API_KEY"]
pinecone_api_key = st.secrets["PINECONE_API_KEY"]

# Streamlit UI
st.set_page_config(page_title="AICA RAG Chatbot", page_icon="🧠")
st.title("AICA RAG Chatbot 🤖")

# Step 1: Initialize Pinecone client (v2.x)
pc = PineconeClient(api_key=pinecone_api_key)

# Step 2: Create or retrieve index
index_name = "aica-chatbot"

if index_name not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Step 3: Get the Index object (IMPORTANT: must use .Index() constructor explicitly)
index = pc.Index(index_name)

# Step 4: Embeddings
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Step 5: LangChain Vectorstore (pass Index object directly)
vectorstore = LangchainPinecone(
    index=index,
    embedding=embeddings,
    text_key="text"
)

# Step 6: LangChain QA pipeline
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(openai_api_key=openai_api_key),
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# Step 7: User input
query = st.text_input("Ask your question:")

if query:
    with st.spinner("Thinking..."):
        result = qa.run(query)
        st.success(result)
