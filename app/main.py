from fastapi import FastAPI
from app.routes import upload, query

app = FastAPI(
    title="ChatPDF API",
    description="RAG-based PDF Question Answering System",
    version="1.0"
)

# Register routes
app.include_router(upload.router, prefix="/api", tags=["Upload"])
app.include_router(query.router, prefix="/api", tags=["Query"])


@app.get("/")
def root():
    return {"message": "ChatPDF API is running 🚀"}