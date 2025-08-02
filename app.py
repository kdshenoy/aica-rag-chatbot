import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# Load .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Streamlit app title
st.title("AICA RAG Chatbot 🤖")

# Initialize Pinecone v3
pc = Pinecone(api_key=pinecone_api_key)
index_name = "aica-chatbot"

# Check if index exists, else create it
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Upload PDF file
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")
query = st.text_input("Ask a question about the PDF")

if uploaded_file and query:
    with st.spinner("Processing..."):

        # Load and split PDF
        loader = PyPDFLoader(uploaded_file.name)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)

        # Embed and store in Pinecone
        embeddings = OpenAIEmbeddings()
        vectorstore = LangchainPinecone.from_documents(
            docs, embedding=embeddings, index_name=index_name
        )

        # Run RetrievalQA chain
        qa = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(temperature=0),
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )

        result = qa.run(query)
        st.success(result)
