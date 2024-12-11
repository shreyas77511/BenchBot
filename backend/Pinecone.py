import os
import chainlit as cl
from flask import Flask, request, jsonify
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone  # Correct import for Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from pinecone import Pinecone as OfficialPinecone, ServerlessSpec  # Official Pinecone client for low-level tasks
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Step 1: Load environment variables from .env file
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

if not OPENAI_API_KEY or not PINECONE_API_KEY or not PINECONE_ENV:
    raise ValueError("API key(s) not found. Please make sure the .env file contains the necessary API keys.")

# Initialize Pinecone instance
pc = OfficialPinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index_name = "quickstart"

# Initialize embeddings model
embeddings_model = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# Ensure the Pinecone index exists
if index_name in pc.list_indexes().names():
    pc.delete_index(index_name)

pc.create_index(
    name=index_name,
    dimension=1536,  # Correct dimension
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)

# Step 2: Load Excel Data and Convert to Documents
def load_excel_to_documents(file_path):
    data = pd.read_excel(file_path, engine="openpyxl")
    if data.empty:
        raise ValueError("The Excel file is empty.")
    
    documents = []
    for idx, row in data.iterrows():
        content = "\n".join([f"{col}: {str(row[col])}" for col in data.columns])
        documents.append(Document(page_content=content, metadata={"source": f"Row {idx + 1}"}))
    
    return documents

# Step 3: Initialize Vectorstore
def initialize_vectorstore(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    split_docs = text_splitter.split_documents(documents)
    
    vectorstore = Pinecone.from_documents(documents=split_docs, embedding=embeddings_model, index_name=index_name)
    return vectorstore

# Load documents and initialize vectorstore
excel_file_path = "./fin_ed_docs/Test3.xlsx"
documents = load_excel_to_documents(excel_file_path)
vectorstore = initialize_vectorstore(documents)

# Set up Prompt Template
prompt_template = """
You are a chatbot designed to answer questions based on employee data stored in a knowledge base.
The data includes:
- Employee ID, Employee Name, Employee Type
- Title as per headcount, Employee Location, Delivery Location
- Employee Status, Billable Category, Department, Sub-department
- Customer Name, Project Name, Roll-off Date, Bench Start Date
- Total Experience, Last Working Day, Vendor Name

When experience is asked please refer the skillsets primary and secondary, with dbiz experience and total experience.

Whenever you are giving date format please mention the month while giving response!

Use the provided context to answer questions accurately and concisely with a brief description. If the question cannot be answered based on the context, say: "I could not find relevant information."

Question: {question}
Context: {context}
"""

prompt = PromptTemplate(input_variables=["question", "context"], template=prompt_template)

# Step 4: Set up Chainlit Bot
@cl.on_chat_start
async def on_chat_start():
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, api_key=OPENAI_API_KEY)
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    
    cl.user_session.set("chain", chain)
    await cl.Message(content="Hi! I am your knowledge base assistant. Ask me anything.").send()

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    if not chain:
        await cl.Message(content="Error: Chain not initialized!").send()
        return

    response = await chain.acall({"question": message.content})
    answer = response["answer"]
    source_docs = response.get("source_documents", [])
    source_text = "\n".join([f"- {doc.metadata['source']}" for doc in source_docs]) if source_docs else "(No sources found)"
    await cl.Message(content=f"{answer}\n\nSources:\n{source_text}").send()

# Step 5: Expose the bot's functionality via Flask
@app.route("/chat", methods=["POST"])
def chat():
    message = request.json.get("message")
    if not message:
        return jsonify({"error": "No message provided"}), 400
    
    chain = cl.user_session.get("chain")
    if not chain:
        return jsonify({"error": "Chain not initialized"}), 500
    
    response = chain.acall({"question": message})
    return jsonify({"answer": response["answer"], "sources": response.get("source_documents", [])})

if __name__ == "__main__":
    import os
    import chainlit as cl

    # Get the PORT from the environment variable (Render provides this)
    PORT = int(os.getenv("PORT", 5000))  # Default to 5000 if PORT is not set

    # Run the Chainlit app, binding to 0.0.0.0 and the correct port
    cl.run(host="0.0.0.0", port=PORT)

