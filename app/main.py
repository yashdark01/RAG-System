# main.py
from fastapi import FastAPI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import routers for document ingestion and querying
from app.api import ingest, query

# Initialize FastAPI application with a title
app = FastAPI(title="RAG System API")

# Include the ingestion router with /api prefix
app.include_router(ingest.router, prefix="/api", tags=["Ingestion"])

# Include the query router with /api prefix
app.include_router(query.router, prefix="/api", tags=["Query"])

# Define GET endpoint for health check
@app.get("/", tags=["Health"])
def health_check():
    return {"status": "RAG System API is running."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)