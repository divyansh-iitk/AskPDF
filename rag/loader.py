import os
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader
from pathlib import Path
from typing import List
from logger import logging



def process_all_pdfs(pdf_directory) -> List[Document]:
    """Reads all pdfs in the directory, adds some metadata

    Args:
        pdf_directory (str)

    Returns:
        List[Document]
    """
    
    all_documents = []
    pdf_dir = Path(pdf_directory)
    
    ## Finding all pdfs recursivly
    pdf_files = list(pdf_dir.glob("**/*.pdf"))
    n = 0
    for pdf_file in pdf_files:
        try:
            loader = PyMuPDFLoader(str(pdf_file))
            documents = loader.load()
            
            ## Assigning aditional metadata
            for doc in documents:
                doc.metadata["source_file"] = pdf_file.name
                doc.metadata["file_type"] = "pdf"
            all_documents.extend(documents)
            n += 1
        except Exception as e:
            print(f"Error: {e}")
    logging.info(f"PDFs loaded: {n}")
    return all_documents
            
    