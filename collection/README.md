# RAG System API - Postman Collection

This Postman collection contains all the available endpoints for the RAG System API.

## Setup

1. Import the `RAG_System_API.postman_collection.json` file into Postman
2. The collection uses a variable `{{base_url}}` which is set to `http://127.0.0.1:8000` by default
3. Make sure your RAG System API is running on the specified base URL

## Available Endpoints

### 1. Health Check
- **Method**: GET
- **URL**: `{{base_url}}/`
- **Description**: Check if the API is running
- **Expected Response**: `{"status": "RAG System API is running."}`

### 2. Upload Document
- **Method**: POST
- **URL**: `{{base_url}}/api/upload`
- **Description**: Upload a PDF document for indexing
- **Body**: Form-data with key `file` (select a PDF file)
- **Expected Response**: `{"message": "filename.pdf uploaded and indexed successfully."}`

### 3. Ask Question
- **Method**: POST
- **URL**: `{{base_url}}/api/ask`
- **Description**: Ask a question based on indexed documents
- **Headers**: `Content-Type: application/json`
- **Body**:
  ```json
  {
    "question": "Your question here"
  }
  ```
- **Expected Response**:
  ```json
  {
    "answer": "The answer based on the documents",
    "sources": [
      {
        "source": "document_name.pdf",
        "page": 1
      }
    ]
  }
  ```

## Usage Flow

1. **Start the API**: Make sure the FastAPI server is running
2. **Check Health**: Use the Health Check endpoint to verify the API is running
3. **Upload Documents**: Use the Upload Document endpoint to index PDF files
4. **Ask Questions**: Use the Ask Question endpoint to query the indexed documents

## Notes

- The upload and ask endpoints require the API routers to be enabled in `main.py`
- Make sure to upload PDF documents before asking questions
- The system uses RAG (Retrieval-Augmented Generation) to provide answers based on the uploaded documents