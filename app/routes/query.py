from fastapi import APIRouter, HTTPException
from rag.chain import GroqLLM
from app.schemas.query import QueryRequest
from logger import logging
from rag.retriever import RAGRetriver
from rag.embeddings import Embedding_manager
from rag.vectorstore import VectorStore

router = APIRouter()


@router.post("/query")
async def query_pdf(request: QueryRequest):
    """
    Query ingested PDFs
    """

    try:
        # Step 1: Retrieve documents
        vector_store = VectorStore()
        embedding_manager = Embedding_manager()
        retriever = RAGRetriver(vector_store, embedding_manager)
        retrieved_docs = retriever.retrieve(request.question)

        if not retrieved_docs:
            return {
                "answer": "No relevant information found.",
                "sources": []
            }

        # Step 2: Generate answer
        llm = GroqLLM()
        answer = llm.generate_response(
            query=request.question,
            retrieved_docs=retrieved_docs
        )

        return {
            "answer": answer,
            "sources": [
                {
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "score": doc["similarity_score"]
                }
                for doc in retrieved_docs
            ],
            "num_sources": len(retrieved_docs)
        }

    except Exception as e:
        logging.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail="Error processing query")