from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi import FastAPI
import os
from rag.ingest import ingest_pdfs
from utils.config import IngestConfig
from logger import logging

router = APIRouter()

UPLOAD_DIR = IngestConfig.pdf_dir
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF and ingest into vector DB
    """

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    try:
        # Save file
        with open(file_path, "wb") as f:
            f.write(await file.read())

        logging.info(f"File saved at {file_path}")

        # Run ingestion
        ingest_pdfs(UPLOAD_DIR)

        return {
            "message": "PDF uploaded and processed successfully",
            "filename": file.filename
        }

    except Exception as e:
        logging.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail="Error processing PDF")