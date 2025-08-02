import os
import streamlit as st
from dotenv import load_dotenv
import tempfile

from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pinecone import Pinecone, ServerlessSpec
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA

# ─── Load Secrets ─────────────────────────────────────
load_dotenv()
openai_api_key = st.secrets["OPENAI_API_KEY"]
pinecone_api_key = st.secrets["PINECONE_API_KEY"]

# ─── Streamlit Page Setup ─────────────────────────────
st.set_page_config(page_title="AICA RAG Chatbot", page_icon="🤖")
st.title("AICA RAG Chatbot")

# ─── Initialize Pinecone ──────────────────────────────
pc = Pinecone(api_key=pinecone_api_key)
index_name = "aica-chatbot"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# ─── Upload PDF and Ask Questions ─────────────────────
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")
query = st.text_input("Ask a question about the uploaded document")

if uploaded_file and query:
    with st.spinner("Processing..."):

        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        # Load and split PDF
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)

        # Embedding model
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

        # Create vectorstore
        vectorstore = LangchainPinecone.from_documents(
            docs, embedding=embeddings, index_name=index_name
        )

        # QA Chain
        qa = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(openai_api_key=openai_api_key, temperature=0),
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )

        # Run Query
        answer = qa.run(query)
        st.success(answer)
