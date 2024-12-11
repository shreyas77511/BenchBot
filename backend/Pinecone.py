import os
import chainlit as cl
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone  # Correct import for Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone as OfficialPinecone, ServerlessSpec  # Official Pinecone client for low-level tasks

# Step 1: Load environment variables from .env file
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

# Ensure environment variables are loaded correctly
if not OPENAI_API_KEY or not PINECONE_API_KEY or not PINECONE_ENV:
    raise ValueError("API key(s) not found. Please make sure the .env file contains the necessary API keys.")

# Step 2: Initialize Pinecone instance
pc = OfficialPinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

index_name = "quickstart"

# Step 3: Initialize embeddings model
embeddings_model = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

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

# Verify index creation
index_info = pc.describe_index(index_name)
if index_info["dimension"] != 1536:
    raise ValueError(f"Index creation failed. Expected dimension 1536 but got {index_info['dimension']}.")

# Step 4: Load Excel Data and Convert to Documents
def load_excel_to_documents(file_path):
    data = pd.read_excel(file_path, engine="openpyxl")
    if data.empty:
        raise ValueError("The Excel file is empty.")
    
    documents = []
    for idx, row in data.iterrows():
        content = "\n".join([f"{col}: {str(row[col])}" for col in data.columns])
        documents.append(Document(page_content=content, metadata={"source": f"Row {idx + 1}"}))
    
    return documents

# Step 5: Initialize Vectorstore
def initialize_vectorstore(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    split_docs = text_splitter.split_documents(documents)
    
    # Use langchain_community.vectorstores.Pinecone's from_documents method
    vectorstore = Pinecone.from_documents(documents=split_docs, embedding=embeddings_model, index_name=index_name)
    return vectorstore

# Step 6: Load Data and Create Vectorstore
excel_file_path = "./fin_ed_docs/Test3.xlsx"
documents = load_excel_to_documents(excel_file_path)
vectorstore = initialize_vectorstore(documents)

# Step 7: Set up Prompt Template
prompt_template = """
You are a chatbot designed to answer questions based on employee data stored in a knowledge base.
The data includes:
- Employee ID, Employee Name, Employee Type
- Title as per headcount, Employee Location, Delivery Location
- Employee Status, Billable Category, Department, Sub-department
- Customer Name, Project Name, Roll-off Date, Bench Start Date
- Total Experience, Last Working Day, Vendor Name

When experience is asked please refer the skillsets primary and  skillsets secondary, with dbiz experience and total experience.

Whenever you are giving date format please mention the month while giving response!

Use the provided context to answer questions accurately and concisely with a brief description. If the question cannot be answered based on the context, say: "I could not find relevant information."

Question: {question}
Context: {context}
"""

prompt = PromptTemplate(input_variables=["question", "context"], template=prompt_template)

# Step 8: Set up Chainlit Bot
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

if __name__ == "__main__":
    # Explicitly set the host and port in environment variables for Render deployment
    os.environ["CHAINLIT_HOST"] = "0.0.0.0"  # Bind to all network interfaces
    os.environ["CHAINLIT_PORT"] = os.getenv("PORT", "5000")  # Use Render's PORT variable or default to 5000
    
    print(f"Binding to host: {os.getenv('CHAINLIT_HOST')}")
    print(f"Binding to port: {os.getenv('CHAINLIT_PORT')}")

    # Start the Chainlit application
    cl.run()
