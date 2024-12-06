# import os
# import pandas as pd
# from langchain_community.embeddings import OpenAIEmbeddings  # Use the correct import
# import chromadb
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.schema import Document  # Import the Document class

# # Initialize Chroma client
# client = chromadb.Client()

# # Create a collection in Chroma to store documents
# collection = client.create_collection(name="bench_candidates")

# # Set your OpenAI API key
# OPENAI_API_KEY= os.getenv("OPENAI_API_KEY")

# # Initialize the OpenAI embeddings model from LangChain
# embeddings_model = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

# # Load Excel Data
# def load_excel_to_documents(file_path):
#     # Read the Excel file
#     data = pd.read_excel(file_path, engine="openpyxl")

#     documents = []
#     for idx, row in data.iterrows():
#         content = "\n".join([f"{col}: {str(row[col])}" for col in data.columns])
#         # Create a Document object with content and metadata
#         document = Document(
#             page_content=content,  # The content of the document
#             metadata={'source': f"Row {idx + 1}"}  # Any additional metadata (e.g., row number)
#         )
#         documents.append(document)
    
#     return documents

# # Initialize vectorstore (Chroma) and insert documents
# def store_in_chroma(documents):
#     # Create a text splitter to break the documents into chunks
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     split_docs = text_splitter.split_documents(documents)

#     # Generate embeddings for each document chunk
#     embeddings = [embeddings_model.embed_document(doc.page_content) for doc in split_docs]

#     # Insert the document and embeddings into Chroma collection
#     for idx, doc in enumerate(split_docs):
#         collection.add(
#             documents=[doc.page_content],
#             metadatas=[doc.metadata],
#             embeddings=[embeddings[idx]],
#             ids=[f"doc_{idx}"]
#         )

#     print("Documents have been stored in Chroma successfully.")

# # Process the uploaded Excel file
# def process_uploaded_file(file_path):
#     documents = load_excel_to_documents(file_path)
#     store_in_chroma(documents)

# # Query Chroma database
# def query_chroma(query_text):
#     # Generate embedding for the query
#     query_embedding = embeddings_model.embed_document(query_text)

#     # Perform the similarity search in Chroma
#     results = collection.query(
#         query_embeddings=[query_embedding],
#         n_results=3  # Number of results to retrieve
#     )

#     # Print results
#     for result in results['documents']:
#         print("Found document:", result)

# # Example Usage

# # 1. Load and store Excel file
# file_path = './fin_ed_docs'
# process_uploaded_file(file_path)

# # 2. Query the Chroma database
# query_text = "What is the skillset of candidate X?"  # Your query text
# query_chroma(query_text)


