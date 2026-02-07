# main.py
import logging
import os
from fastapi import FastAPI
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Import routers for document ingestion and querying
from app.api import ingest, query

# Initialize FastAPI application with a title
app = FastAPI(title="RAG System API")


@app.on_event("startup")
async def startup_event():
    """Validate environment variables and log startup information."""
    logger.info("Starting RAG System API...")
    
    # Check for required environment variables
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        logger.warning(
            "GROQ_API_KEY not found in environment variables. "
            "The /api/ask endpoint will fail. "
            "Get your free API key from https://console.groq.com/keys"
        )
    else:
        logger.info("GROQ_API_KEY found in environment variables")
    
    logger.info("RAG System API started successfully")

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