# app/services/retriever.py

def get_retriever(vector_db, k=4):
    """
    Create a retriever from the vector database.

    Args:
        vector_db (FAISS): The FAISS vector database.
        k (int): The number of top documents to retrieve. Default is 4.
    Returns:
        Retriever: A retriever object for fetching relevant documents.
    """
    return vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )