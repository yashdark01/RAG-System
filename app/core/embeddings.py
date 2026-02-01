# app/core/embeddings.py
from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_model():
    """
    Get the HuggingFace embeddings model.
    This model is completely free to use.

    Returns:
        HuggingFaceEmbeddings: The embeddings model instance.
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},  # Use 'cuda' if GPU is available
        encode_kwargs={'normalize_embeddings': True}
    )


def get_embeddings(documents):
    """
    Generate embeddings for a list of documents using HuggingFace embeddings.

    Args:
        documents (List[Document]): A list of Document objects to generate embeddings for.
    Returns:
        List[List[float]]: A list of embeddings corresponding to the input documents.
    """
    embeddings_model = get_embedding_model()
    texts = [doc.page_content for doc in documents]
    embeddings = embeddings_model.embed_documents(texts)
    return embeddings