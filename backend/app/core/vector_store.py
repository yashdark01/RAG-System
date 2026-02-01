# app/core/vector_store.py
from langchain_community.vectorstores import FAISS
import os

VECTOR_DB_PATH = "data/vector_db"  # Path to store the vector database


def create_vector_store(documents, embedding_model):
    """
    Create and save a FAISS vector store from documents and their embedding model.

    Args:
        documents (List[Document]): A list of Document objects.
        embedding_model: The embedding model to use.
    """
    if not documents:
        raise ValueError("No documents provided to create vector store")
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(VECTOR_DB_PATH), exist_ok=True)
    
    # Check if vector store already exists
    if os.path.exists(VECTOR_DB_PATH):
        # Load existing store and add new documents
        existing_store = FAISS.load_local(
            VECTOR_DB_PATH,
            embedding_model,
            allow_dangerous_deserialization=True
        )
        existing_store.add_documents(documents)
        existing_store.save_local(VECTOR_DB_PATH)
    else:
        # Create new vector store
        vector_store = FAISS.from_documents(documents, embedding_model)
        vector_store.save_local(VECTOR_DB_PATH)


def load_vector_store(embeddings):
    """
    Load the FAISS vector store from disk.

    Args:
        embeddings (Embeddings): The embeddings model to use for the vector store.
    Returns:
        FAISS: The loaded FAISS vector store.
    Raises:
        FileNotFoundError: If the vector store doesn't exist.
    """
    if not os.path.exists(VECTOR_DB_PATH):
        raise FileNotFoundError(
            f"Vector store not found at {VECTOR_DB_PATH}. "
            "Please upload documents first using the /api/upload endpoint."
        )
    
    vector_store = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vector_store