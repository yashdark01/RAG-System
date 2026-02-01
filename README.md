# RAG System

A Retrieval-Augmented Generation (RAG) system backend built with FastAPI and LangChain for document-based question answering.

## Features

- **Document Ingestion**: Upload and index PDF documents using FAISS vector store and sentence transformers.
- **Question Answering**: Ask questions and get AI-generated answers based on indexed documents with source references.
- **Multiple LLM Support**: Integrates with OpenAI, Google Gemini, Ollama, and Hugging Face models.
- **FastAPI Backend**: RESTful API with automatic documentation.
- **Vector Search**: Efficient similarity search using FAISS for document retrieval.

## Tech Stack

- **Backend**: FastAPI, Uvicorn
- **AI/ML**: LangChain, Sentence Transformers, FAISS
- **LLMs**: OpenAI GPT, Google Gemini, Ollama, Hugging Face Transformers
- **Document Processing**: PyPDF
- **Environment**: Python 3.11+

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yashdark01/RAG-System.git
   cd RAG-System
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   Create a `.env` file in the root directory with your API keys:
   ```env
   OPENAI_API_KEY=your_openai_key
   GOOGLE_API_KEY=your_google_key
   # Add other keys as needed
   ```

## Usage

1. Start the server:
   ```bash
   python app/main.py
   ```
   The API will be available at `http://127.0.0.1:8000`

2. Access the API documentation at `http://127.0.0.1:8000/docs`

## Docker Usage

1. Ensure Docker and Docker Compose are installed.

2. Build and run the application using Docker Compose:
   ```bash
   docker-compose up --build
   ```
   The API will be available at `http://127.0.0.1:8000`

3. To run in the background:
   ```bash
   docker-compose up -d --build
   ```

4. Stop the containers:
   ```bash
   docker-compose down
   ```

Note: Data (vector database and uploaded documents) is persisted in the `data/` directory via Docker volumes. Ensure your `.env` file is in the root directory with required API keys.

## API Endpoints

### Health Check
- **GET** `/`
- Returns the status of the API.

### Upload Document
- **POST** `/api/upload`
- Upload a PDF file for indexing.
- **Body**: Form data with `file` (PDF file)

### Ask Question
- **POST** `/api/ask`
- Ask a question based on indexed documents.
- **Body**:
  ```json
  {
    "question": "What is the main topic?"
  }
  ```
- **Response**:
  ```json
  {
    "answer": "The main topic is...",
    "sources": [
      {
        "page": 1,
        "source": "document.pdf"
      }
    ]
  }
  ```

## Configuration

The system uses the following components (configurable in `app/core/`):

- **Embeddings**: Sentence Transformers (default: `all-MiniLM-L6-v2`)
- **Vector Store**: FAISS
- **LLM**: Configurable via environment (OpenAI by default)
- **Chunking**: Recursive text splitter with configurable chunk size

## Project Structure

```
RAG-System/
├── app/
│   ├── api/
│   │   ├── ingest.py      # Document upload endpoint
│   │   └── query.py       # Question answering endpoint
│   ├── core/
│   │   ├── embeddings.py  # Embedding configuration
│   │   ├── llm.py         # LLM configuration
│   │   └── vector_store.py # Vector store setup
│   ├── services/
│   │   ├── chunker.py     # Text chunking
│   │   ├── loader.py      # Document loading
│   │   ├── rag_chain.py   # RAG chain setup
│   │   └── retriever.py   # Document retrieval
│   └── main.py            # FastAPI app
├── data/
│   ├── documents/         # Uploaded documents
│   └── vector_db/         # FAISS index
├── collection/            # Postman collection
├── .env                   # Environment variables
├── pyproject.toml         # Project metadata
├── requirements.txt       # Dependencies
└── README.md              # This file
```

## Development

- Run tests: `pytest` (if tests are added)
- Lint code: `flake8` or your preferred linter
- Format code: `black` or your preferred formatter

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Contact

Yash Patidar - yashpatidar9691@gmail.com