#!/usr/bin/env python3
"""
Seed initial knowledge base documents into the RAG vector store.
Can run standalone using DocumentService.
"""

import asyncio
from app.dependencies import dependencies

SAMPLE_DOCUMENTS = [
    {
        "filename": "rag_architecture.md",
        "content": """# RAG (Retrieval-Augmented Generation) Architecture
Retrieval-Augmented Generation (RAG) is an architectural pattern that combines
information retrieval mechanisms with deep generative models to answer questions
grounded in private or domain-specific knowledge bases.

## Components of Modern RAG:
1. **Document Processing & Chunking**: Recursive character text splitting and parent-child hierarchical chunking.
2. **Hybrid Retrieval**: Combining dense semantic vector representations with sparse BM25 keyword matching using Reciprocal Rank Fusion (RRF).
3. **Reranking**: Cross-encoders evaluate query-document pairs to maximize semantic relevancy before feeding context to the LLM.
4. **Agentic Orchestration**: Multi-agent workflows decompose queries, synthesize multi-source answers, and fact-check citations.
""",
        "metadata": {"category": "architecture", "tags": ["rag", "llm", "system-design"]}
    },
    {
        "filename": "agentic_workflows.md",
        "content": """# Multi-Agent Workflow Patterns
Autonomous agentic systems operate beyond standard single-turn retrieval by employing
specialized, cooperating agents:

- **Query Agent**: Analyzes intent, disambiguates queries, and generates sub-queries.
- **Research Agent**: Collects evidence across internal vector stores and Tavily web search.
- **Synthesis Agent**: Merges disparate findings into a cohesive, cited answer.
- **Validation Agent**: Cross-references factual claims against original source text to prevent hallucinations.
""",
        "metadata": {"category": "agents", "tags": ["multi-agent", "agents", "langchain"]}
    }
]

async def seed():
    print("🌱 Starting knowledge base document ingestion...")
    doc_service = dependencies.get_document_service()
    
    for doc in SAMPLE_DOCUMENTS:
        filename = doc["filename"]
        content_bytes = doc["content"].encode("utf-8")
        metadata = doc["metadata"]
        
        print(f"📄 Indexing: {filename}...")
        result = await doc_service.process_and_index(
            content=content_bytes,
            filename=filename,
            metadata=metadata
        )
        print(f"   ✓ Indexed: {result.get('chunks_count', 0)} chunks (ID: {result.get('document_id')})")

    print("🎉 Knowledge base seeded successfully!")

if __name__ == "__main__":
    asyncio.run(seed())

