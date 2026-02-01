# app/services/rag_chain.py
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def build_rag_chain(llm, retriever):
    """
    Build a Retrieval-Augmented Generation (RAG) chain using the provided language model and retriever.
    
    Args:
        llm: The language model to use for generating responses.
        retriever: The retriever to retrieve relevant documents from.
    Returns:
        Runnable: A RAG chain that can be used to answer questions based on retrieved documents.
    """

    template = """Answer the question based only on the following context. If you cannot answer the question based on the context, say "I don't have enough information to answer this question."

Context:
{context}

Question: {question}

Answer: """

    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        """Format retrieved documents into a single string."""
        return "\n\n".join([doc.page_content for doc in docs])

    # Build the chain with proper retriever integration
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain