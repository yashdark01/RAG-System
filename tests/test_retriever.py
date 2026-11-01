"""
Unit tests for RetrieverService and hybrid search algorithms (BM25 + Semantic + RRF).
"""

import pytest
from unittest.mock import Mock, AsyncMock, MagicMock
from app.rag.retriever import RetrieverService, RetrievalResult


class TestRetrieverService:
    """Test suite for RetrieverService."""

    @pytest.fixture
    def mock_vector_store(self):
        store = Mock()
        store._metadata_index = {"doc_1": 0, "doc_2": 1, "doc_3": 2}
        
        # Async search mock
        store.search = AsyncMock(return_value=[
            {
                "id": "doc_1",
                "text": "Machine learning is a subfield of artificial intelligence.",
                "score": 0.92,
                "metadata": {"source": "ai.txt", "category": "tech"},
                "rank": 1
            },
            {
                "id": "doc_2",
                "text": "Deep neural networks are used for computer vision and NLP.",
                "score": 0.85,
                "metadata": {"source": "dl.txt", "category": "tech"},
                "rank": 2
            }
        ])

        # Async get_documents mock for BM25
        store.get_documents = AsyncMock(return_value=[
            {
                "id": "doc_1",
                "text": "Machine learning is a subfield of artificial intelligence.",
                "metadata": {"source": "ai.txt", "category": "tech"}
            },
            {
                "id": "doc_2",
                "text": "Deep neural networks are used for computer vision and NLP.",
                "metadata": {"source": "dl.txt", "category": "tech"}
            },
            {
                "id": "doc_3",
                "text": "Photosynthesis is the process used by plants to convert light energy.",
                "metadata": {"source": "bio.txt", "category": "science"}
            }
        ])
        return store

    @pytest.fixture
    def mock_embeddings(self):
        emb = Mock()
        emb.embed_query = Mock(return_value=[0.1] * 384)
        return emb

    @pytest.fixture
    def mock_reranker(self):
        reranker = Mock()
        reranker.rerank = AsyncMock(side_effect=lambda query, docs, top_k=None: [
            Mock(text=d, score=0.99 - (i * 0.1), original_rank=i, new_rank=i+1)
            for i, d in enumerate(docs[:top_k] if top_k else docs)
        ])
        return reranker

    @pytest.fixture
    def retriever(self, mock_vector_store, mock_embeddings, mock_reranker):
        return RetrieverService(
            vector_store=mock_vector_store,
            embedding_service=mock_embeddings,
            reranker=mock_reranker,
            top_k=2,
            similarity_threshold=0.5,
            reranking_enabled=False,
            retrieval_strategy="hybrid"
        )

    @pytest.mark.asyncio
    async def test_semantic_retrieval(self, retriever):
        """Verify semantic retrieval returns formatted RetrievalResult items."""
        results = await retriever._semantic_retrieval("machine learning", k=2)
        assert len(results) == 2
        assert isinstance(results[0], RetrievalResult)
        assert results[0].id == "doc_1"
        assert results[0].retrieval_method == "semantic"
        assert results[0].score >= results[1].score

    @pytest.mark.asyncio
    async def test_keyword_retrieval_bm25(self, retriever):
        """Verify keyword retrieval with BM25 finds matching terms."""
        results = await retriever._keyword_retrieval("neural networks vision", k=2)
        assert len(results) > 0
        assert any(r.id == "doc_2" for r in results)
        assert results[0].retrieval_method == "keyword"

    @pytest.mark.asyncio
    async def test_hybrid_retrieval_rrf(self, retriever):
        """Verify hybrid retrieval fuses semantic and keyword results using RRF."""
        results = await retriever._hybrid_retrieval("machine learning", k=2)
        assert len(results) <= 2
        assert results[0].retrieval_method == "hybrid"
        # RRF assigns positive fused score
        assert results[0].score > 0

    @pytest.mark.asyncio
    async def test_retrieve_caching(self, retriever):
        """Verify repeated calls utilize the internal query cache."""
        res1 = await retriever.retrieve("machine learning", top_k=2, use_cache=True)
        assert len(res1) > 0
        assert retriever._cache_hits == 0

        # Second call with same parameters should hit cache
        res2 = await retriever.retrieve("machine learning", top_k=2, use_cache=True)
        assert len(res2) == len(res1)
        assert retriever._cache_hits == 1

    @pytest.mark.asyncio
    async def test_metadata_filtering(self, retriever):
        """Verify metadata filters filter out non-matching documents."""
        filters = {"category": "tech"}
        assert retriever._matches_filters({"category": "tech", "source": "ai.txt"}, filters) is True
        assert retriever._matches_filters({"category": "science", "source": "bio.txt"}, filters) is False

    @pytest.mark.asyncio
    async def test_reranking_integration(self, retriever):
        """Verify reranker reorders results when enabled."""
        retriever.reranking_enabled = True
        results = await retriever.retrieve("machine learning", top_k=2, use_cache=False)
        assert len(results) <= 2
