from rag.loader import  process_pdf
from rag.splitter import split_documents
from rag.embeddings import Embedding_manager
from rag.vectorstore import VectorStore
from rag.BM25 import BM25_retriever
from langchain_community.retrievers import BM25Retriever
from logger import logging
from langchain_core.documents import Document


#---------------------#

def ingest_pdf(pdf_file: str, embedding_manager: Embedding_manager, vector_store: VectorStore, flag: bool = True) -> BM25Retriever:
    documents = process_pdf(pdf_file)
    chunks = split_documents(documents)
    if flag:
        texts = [doc.page_content for doc in chunks]
        embeddings = embedding_manager.create_embeddings(texts)

        vector_store.add_documents(documents=chunks, embeddings=embeddings)

        logging.info(f"PDFs ingested into vector DB")
    else:
        logging.info(f"PDF is already ingested into vector DB")
    ##Creating BM25 Retriever
    try:
        collection = vector_store.collection
        all_data = collection.get()
        docs = [
            Document(
                page_content=text,
                metadata=metadata or {}
            )
            for text, metadata in zip(all_data.get("documents"), all_data.get("metadatas"))
        ]
        BM25retriever = BM25_retriever(docs)
        logging.info(f"BM25 retriever build successfully")
    except Exception as e:
        logging.error(f"Error while building BM25 retriever")
        raise
    return BM25retriever

#----------------------#
