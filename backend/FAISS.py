import pandas as pd
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import chainlit as cl
from langchain.chat_models import ChatOpenAI  # Add this import statement

import os
from dotenv import load_dotenv

# Step 1: Load environment variables from .env file
load_dotenv()  # This will load variables from a .env file into environment variables

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY is None:
    raise ValueError("API key not found. Please make sure the .env file contains the OPENAI_API_KEY.")

# Step 2: Load and Process Dynamic Excel Data
def load_excel_to_documents(file_path):
    # Read the Excel file dynamically
    data = pd.read_excel(file_path, engine="openpyxl")
    
    # Ensure there's data
    if data.empty:
        raise ValueError("The Excel file is empty.")
    
    documents = []
    for idx, row in data.iterrows():
        # Create a content string for the document
        content = "\n".join([f"{col}: {str(row[col])}" for col in data.columns])
        
        # Add Document object with dynamic content
        documents.append(Document(page_content=content, metadata={"source": f"Row {idx + 1}"}))
    
    return documents

# Step 3: Initialize Vectorstore
def initialize_vectorstore(documents):
    # Initialize embeddings model using the API key from the environment variable
    embeddings_model = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    
    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(documents)
    
    # Create FAISS vectorstore
    vectorstore = FAISS.from_documents(documents=split_docs, embedding=embeddings_model)
    return vectorstore

# Step 4: Load Data and Create Vectorstore
excel_file_path = "./fin_ed_docs/Test3.xlsx"  # Correct file path
documents = load_excel_to_documents(excel_file_path)
vectorstore = initialize_vectorstore(documents)

# Step 5: Set up Chainlit Bot
@cl.on_chat_start
async def on_chat_start():
    # Initialize conversation memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Set up retrieval-based chain using the API key from the environment variable
    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name="gpt-4", temperature=0, api_key=OPENAI_API_KEY),
        retriever=vectorstore.as_retriever(),
        memory=memory,
    )
    cl.user_session.set("chain", chain)
    await cl.Message(content="Hi! I am your knowledge base assistant. Ask me anything.").send()

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    if not chain:
        await cl.Message(content="Error: Chain not initialized!").send()
        return

    # Process user input
    response = await chain.acall({"question": message.content})
    answer = response["answer"]
    source_docs = response.get("source_documents", [])

    # Include sources in the response if available
    source_text = "\n".join([f"- {doc.metadata['source']}" for doc in source_docs])
    answer += f"\n\nSources:\n{source_text}" if source_docs else "\n\n(No sources found)"

    await cl.Message(content=answer).send()
