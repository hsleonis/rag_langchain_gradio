from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


# Read documents in directory
def read_doc(directory: str):
    file_loader = PyPDFDirectoryLoader(directory)
    documents = file_loader.load()
    return documents


# Divide document into text chunks
def chunk_data(docs, chunk_size=800, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(docs)
    return chunks


# Cosine similarity to retrieve results from vectorDB
def retrieve_query(index, query, k=2):
    matching_results = index.similarity_search(query, k=k)
    return matching_results


# Search answers from Pinecone VectorDB
def retrieve_answers(index, chain, query):
    doc_search = retrieve_query(index, query)
    response = chain.run(input_documents=doc_search, question=query)
    return response


# Get results from query
def qa_manager(query):
    return retrieve_answers(query)
