import os
from pathlib import Path

list_of_files = [
    
    "app/__init__.py",
    "app/main.py",
    "app/routes/upload.py",
    "app/routes/query.py",
    "app/schemas/schema.py",
    
    "rag/__init__.py",
    "rag/loader.py",
    "rag/splitter.py",
    "rag/embeddings.py",
    "rag/vectorstore.py",
    "rag/retriever.py",
    "rag/chain.py",
    
    "utils/__init__.py",
    "utils/config.py",
    
    "frontend/app.py"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
    else:
        print(f"file is already present at: {filepath}")