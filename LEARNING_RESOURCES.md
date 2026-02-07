# Learning Resources for RAG System

Comprehensive documentation links, tutorials, and learning paths for all technologies used in this RAG (Retrieval-Augmented Generation) system.

---

## 📚 Quick Reference Table

| Technology | Purpose | Official Docs | Tutorial |
|------------|---------|---------------|----------|
| **FastAPI** | Web Framework | [fastapi.tiangolo.com](https://fastapi.tiangolo.com/) | [Tutorial](https://fastapi.tiangolo.com/tutorial/) |
| **LangChain** | RAG Framework | [docs.langchain.com](https://docs.langchain.com/) | [Academy](https://academy.langchain.com/) |
| **FAISS** | Vector Database | [GitHub Wiki](https://github.com/facebookresearch/faiss/wiki) | [Tutorial](https://github.com/facebookresearch/faiss/wiki/Getting-started) |
| **Sentence Transformers** | Embeddings | [sbert.net](https://www.sbert.net/) | [Examples](https://www.sbert.net/examples/applications/) |
| **Groq** | LLM Provider | [console.groq.com/docs](https://console.groq.com/docs) | [Quickstart](https://console.groq.com/docs/quickstart) |
| **PyPDF** | PDF Processing | [PyPDF Docs](https://pypdf.readthedocs.io/) | [GitHub](https://github.com/py-pdf/pypdf) |
| **Pydantic** | Data Validation | [docs.pydantic.dev](https://docs.pydantic.dev/) | [Tutorial](https://docs.pydantic.dev/latest/tutorial/) |
| **Uvicorn** | ASGI Server | [uvicorn.org](https://www.uvicorn.org/) | [Deployment](https://www.uvicorn.org/deployment/) |
| **HuggingFace** | ML Models Hub | [huggingface.co](https://huggingface.co/) | [Course](https://huggingface.co/learn/nlp-course/) |

---

## 🎯 Technology-Specific Learning Resources

### 1. FastAPI (Backend Framework)

**What it does in your system:** Powers the REST API endpoints for document upload and querying.

**Official Documentation:**
- **Main Docs**: https://fastapi.tiangolo.com/
- **Tutorial**: https://fastapi.tiangolo.com/tutorial/
- **Advanced User Guide**: https://fastapi.tiangolo.com/advanced/
- **Deployment**: https://fastapi.tiangolo.com/deployment/

**Key Topics to Learn:**
- [Request/Response Models](https://fastapi.tiangolo.com/tutorial/body/)
- [File Uploads](https://fastapi.tiangolo.com/tutorial/request-files/)
- [Dependency Injection](https://fastapi.tiangolo.com/tutorial/dependencies/)
- [Error Handling](https://fastapi.tiangolo.com/tutorial/handling-errors/)
- [API Documentation](https://fastapi.tiangolo.com/tutorial/metadata/)

**Recommended Learning Path:**
1. Start with the [Tutorial - User Guide](https://fastapi.tiangolo.com/tutorial/)
2. Practice with [FastAPI Examples](https://github.com/tiangolo/fastapi/tree/master/docs_src)
3. Learn [Async/Await](https://fastapi.tiangolo.com/async/)
4. Study [Background Tasks](https://fastapi.tiangolo.com/tutorial/background-tasks/)

**Video Courses:**
- [FastAPI Full Course - freeCodeCamp](https://www.youtube.com/watch?v=0sOvCWFmrtA)
- [FastAPI Tutorial - Corey Schafer](https://www.youtube.com/watch?v=7t2alSnE2-I)

---

### 2. LangChain (RAG Framework)

**What it does in your system:** Orchestrates document loading, chunking, embedding, retrieval, and RAG chain construction.

**Official Documentation:**
- **Main Docs**: https://docs.langchain.com/
- **Python API Reference**: https://api.python.langchain.com/
- **LangChain Academy** (Interactive Courses): https://academy.langchain.com/
- **LangSmith** (Observability): https://docs.smith.langchain.com/

**Core Concepts:**
- **Document Loaders**: https://python.langchain.com/docs/how_to/document_loaders/
- **Text Splitters**: https://python.langchain.com/docs/how_to/text_splitters/
- **Vector Stores**: https://python.langchain.com/docs/how_to/vectorstores/
- **Retrievers**: https://python.langchain.com/docs/how_to/retrievers/
- **RAG Chains**: https://python.langchain.com/docs/use_cases/question_answering/

**Essential Tutorials:**
1. **[Build a RAG Agent](https://python.langchain.com/docs/tutorials/retrievers/)** - Complete Q&A application
2. **[Semantic Search Engine](https://docs.langchain.com/oss/python/langchain/knowledge-base)** - Document search implementation
3. **[RAG Tutorial](https://docs.langchain.com/oss/python/langchain/rag)** - End-to-end RAG system
4. **[Retrieval Guide](https://docs.langchain.com/oss/python/langchain/retrieval)** - Deep dive into retrieval

**Advanced RAG Course (Recommended):**
- **LangChain Tutorials - RAG Applications**: https://langchain-tutorials.com/lessons/rag-applications
  - 35 lessons, ~10 hours total
  - Covers: Vector embeddings, Vector databases (FAISS, ChromaDB, Pinecone), Advanced RAG techniques, Multi-query retrieval, Reranking, Hybrid search, RAG evaluation with RAGAS

**Key Modules Used in Your System:**
- `langchain_community.document_loaders` - PDF loading
- `langchain_text_splitters` - Document chunking
- `langchain_community.vectorstores` - FAISS integration
- `langchain_core.runnables` - Chain construction
- `langchain_core.prompts` - Prompt templates

**Learning Path:**
1. Start with [LangChain Quickstart](https://python.langchain.com/docs/get_started/introduction)
2. Learn [Document Loaders](https://python.langchain.com/docs/how_to/document_loaders/)
3. Master [Vector Stores & Retrievers](https://python.langchain.com/docs/how_to/retrievers/)
4. Build [RAG Applications](https://python.langchain.com/docs/use_cases/question_answering/)
5. Take the [Advanced RAG Course](https://langchain-tutorials.com/lessons/rag-applications)

---

### 3. FAISS (Vector Database)

**What it does in your system:** Stores document embeddings and performs fast similarity search for retrieval.

**Official Documentation:**
- **GitHub Repository**: https://github.com/facebookresearch/faiss
- **Wiki Documentation**: https://github.com/facebookresearch/faiss/wiki
- **Getting Started**: https://github.com/facebookresearch/faiss/wiki/Getting-started
- **Index Types**: https://github.com/facebookresearch/faiss/wiki/Faiss-indexes

**Key Concepts:**
- **Index Types**: Flat, IVF, HNSW, LSH
- **Similarity Metrics**: L2 distance, Inner product, Cosine similarity
- **GPU Support**: https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU

**Tutorials:**
- [FAISS Tutorial - Pinecone](https://www.pinecone.io/learn/faiss-tutorial/)
- [Vector Similarity Search with FAISS](https://www.pinecone.io/learn/what-is-similarity-search/)
- [FAISS Python API](https://github.com/facebookresearch/faiss/wiki/Faiss-code-structure)

**LangChain Integration:**
- [LangChain FAISS Documentation](https://python.langchain.com/docs/integrations/vectorstores/faiss)
- [FAISS Retriever Guide](https://python.langchain.com/docs/how_to/retrievers/)

**Learning Path:**
1. Read [Getting Started Guide](https://github.com/facebookresearch/faiss/wiki/Getting-started)
2. Understand [Index Types](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)
3. Learn [LangChain Integration](https://python.langchain.com/docs/integrations/vectorstores/faiss)
4. Practice with [Similarity Search Tutorials](https://www.pinecone.io/learn/faiss-tutorial/)

---

### 4. Sentence Transformers / HuggingFace (Embeddings)

**What it does in your system:** Converts text documents into vector embeddings for semantic search.

**Sentence Transformers Documentation:**
- **Official Site**: https://www.sbert.net/
- **GitHub**: https://github.com/UKPLab/sentence-transformers
- **Documentation**: https://www.sbert.net/docs/
- **Pre-trained Models**: https://www.sbert.net/docs/pretrained_models.html

**HuggingFace Resources:**
- **HuggingFace Hub**: https://huggingface.co/sentence-transformers
- **Model Hub**: https://huggingface.co/models?library=sentence-transformers
- **NLP Course**: https://huggingface.co/learn/nlp-course/
- **Documentation**: https://huggingface.co/docs/transformers/

**Key Topics:**
- [Semantic Similarity](https://www.sbert.net/docs/usage/semantic_textual_similarity.html)
- [Embedding Models](https://www.sbert.net/docs/pretrained_models.html)
- [Fine-tuning](https://www.sbert.net/docs/training/overview.html)
- [Model Comparison](https://www.sbert.net/docs/pretrained_models.html#model-overview)

**LangChain Integration:**
- [HuggingFace Embeddings](https://python.langchain.com/docs/integrations/text_embedding/huggingface)
- [Sentence Transformers Guide](https://python.langchain.com/docs/integrations/text_embedding/sentence_transformers)

**Model Used in Your System:**
- `all-MiniLM-L6-v2` - Fast, efficient model for semantic search
- [Model Card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

**Alternative Models to Explore:**
- `all-mpnet-base-v2` - Better quality, slower
- `multi-qa-MiniLM-L6-cos-v1` - Optimized for Q&A
- `paraphrase-multilingual-MiniLM-L12-v2` - Multilingual support

**Learning Path:**
1. Read [Sentence Transformers Documentation](https://www.sbert.net/docs/)
2. Explore [Pre-trained Models](https://www.sbert.net/docs/pretrained_models.html)
3. Learn [Semantic Similarity](https://www.sbert.net/docs/usage/semantic_textual_similarity.html)
4. Study [LangChain Integration](https://python.langchain.com/docs/integrations/text_embedding/huggingface)
5. Try [Fine-tuning Tutorial](https://www.sbert.net/docs/training/overview.html)

---

### 5. Groq (LLM Provider)

**What it does in your system:** Provides fast LLM inference for generating answers from retrieved documents.

**Official Documentation:**
- **API Documentation**: https://console.groq.com/docs
- **Quickstart Guide**: https://console.groq.com/docs/quickstart
- **Models Available**: https://console.groq.com/docs/models
- **Console**: https://console.groq.com/

**LangChain Integration:**
- **LangChain Groq Docs**: https://python.langchain.com/docs/integrations/llms/groq
- **ChatGroq Class**: https://api.python.langchain.com/en/latest/llms/langchain_groq.chat_models.ChatGroq.html

**Available Models:**
- `llama-3.1-8b-instant` - Fast, efficient
- `llama-3.1-70b-versatile` - High quality
- `mixtral-8x7b-32768` - Large context window
- `gemma2-9b-it` - Google's Gemma model

**Key Features:**
- Fast inference (low latency)
- Free tier available
- REST API
- Streaming support

**Learning Resources:**
- [Groq API Reference](https://console.groq.com/docs/api-reference)
- [Rate Limits](https://console.groq.com/docs/rate-limits)
- [Error Handling](https://console.groq.com/docs/errors)

**Learning Path:**
1. Sign up at [console.groq.com](https://console.groq.com/)
2. Read [Quickstart Guide](https://console.groq.com/docs/quickstart)
3. Learn [LangChain Integration](https://python.langchain.com/docs/integrations/llms/groq)
4. Explore [Available Models](https://console.groq.com/docs/models)

---

### 6. PyPDF (PDF Processing)

**What it does in your system:** Extracts text content from uploaded PDF documents.

**Official Documentation:**
- **Documentation**: https://pypdf.readthedocs.io/
- **GitHub**: https://github.com/py-pdf/pypdf
- **PyPI**: https://pypi.org/project/pypdf/

**Key Features:**
- PDF text extraction
- PDF metadata extraction
- PDF manipulation
- Page-by-page reading

**LangChain Integration:**
- [PyPDFLoader](https://python.langchain.com/docs/integrations/document_loaders/pypdf)

**Tutorials:**
- [PyPDF Documentation](https://pypdf.readthedocs.io/en/latest/user/extract-text.html)
- [Working with PDFs in Python](https://realpython.com/pdf-python/)

**Learning Path:**
1. Read [PyPDF Documentation](https://pypdf.readthedocs.io/)
2. Learn [Text Extraction](https://pypdf.readthedocs.io/en/latest/user/extract-text.html)
3. Explore [LangChain PyPDFLoader](https://python.langchain.com/docs/integrations/document_loaders/pypdf)

---

### 7. Pydantic (Data Validation)

**What it does in your system:** Validates request/response models for API endpoints.

**Official Documentation:**
- **Main Docs**: https://docs.pydantic.dev/
- **Tutorial**: https://docs.pydantic.dev/latest/tutorial/
- **API Reference**: https://docs.pydantic.dev/latest/api/

**Key Concepts:**
- [Models](https://docs.pydantic.dev/latest/concepts/models/)
- [Field Validation](https://docs.pydantic.dev/latest/concepts/validators/)
- [Serialization](https://docs.pydantic.dev/latest/concepts/serialization/)
- [Settings Management](https://docs.pydantic.dev/latest/concepts/settings/)

**FastAPI Integration:**
- [FastAPI + Pydantic](https://fastapi.tiangolo.com/tutorial/body/)
- [Request Models](https://fastapi.tiangolo.com/tutorial/body/#request-body-with-parameters)

**Learning Path:**
1. Complete [Pydantic Tutorial](https://docs.pydantic.dev/latest/tutorial/)
2. Learn [Model Validation](https://docs.pydantic.dev/latest/concepts/models/)
3. Study [FastAPI Integration](https://fastapi.tiangolo.com/tutorial/body/)

---

### 8. Uvicorn (ASGI Server)

**What it does in your system:** Runs the FastAPI application as a production server.

**Official Documentation:**
- **Main Docs**: https://www.uvicorn.org/
- **Deployment**: https://www.uvicorn.org/deployment/
- **Settings**: https://www.uvicorn.org/settings/

**Key Topics:**
- [Running Uvicorn](https://www.uvicorn.org/#running-uvicorn)
- [Deployment Options](https://www.uvicorn.org/deployment/)
- [Performance Tuning](https://www.uvicorn.org/settings/#performance)

**FastAPI Integration:**
- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)

**Learning Path:**
1. Read [Uvicorn Documentation](https://www.uvicorn.org/)
2. Learn [Deployment Options](https://www.uvicorn.org/deployment/)
3. Study [FastAPI Deployment Guide](https://fastapi.tiangolo.com/deployment/)

---

### 9. Python-dotenv (Environment Management)

**What it does in your system:** Loads environment variables from `.env` file for API keys and configuration.

**Official Documentation:**
- **GitHub**: https://github.com/theskumar/python-dotenv
- **PyPI**: https://pypi.org/project/python-dotenv/
- **Documentation**: https://saurabh-kumar.com/python-dotenv/

**Key Features:**
- Load `.env` files
- Environment variable management
- Secure configuration handling

**Best Practices:**
- Never commit `.env` files
- Use `.env.example` as template
- Document required variables

**Learning Path:**
1. Read [python-dotenv GitHub](https://github.com/theskumar/python-dotenv)
2. Learn [Environment Variables Best Practices](https://12factor.net/config)

---

## 🎓 Recommended Learning Paths

### Path 1: Beginner - Building Your First RAG System

**Duration:** 2-3 weeks

1. **Week 1: Foundations**
   - Learn Python basics (if needed)
   - Study FastAPI: [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)
   - Understand REST APIs
   - Practice with [FastAPI Examples](https://github.com/tiangolo/fastapi/tree/master/docs_src)

2. **Week 2: LangChain & RAG**
   - Complete [LangChain Quickstart](https://python.langchain.com/docs/get_started/introduction)
   - Follow [RAG Tutorial](https://python.langchain.com/docs/use_cases/question_answering/)
   - Build [Semantic Search Engine](https://docs.langchain.com/oss/python/langchain/knowledge-base)
   - Understand embeddings and vector stores

3. **Week 3: Integration & Deployment**
   - Integrate all components
   - Learn FAISS basics
   - Deploy with Uvicorn
   - Test your RAG system

**Resources:**
- [LangChain Academy](https://academy.langchain.com/) - Free courses
- [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/) - Complete guide

---

### Path 2: Intermediate - Production-Ready RAG

**Duration:** 4-6 weeks

1. **Advanced LangChain**
   - Complete [Advanced RAG Course](https://langchain-tutorials.com/lessons/rag-applications)
   - Learn MMR retrieval
   - Implement hybrid search
   - Add re-ranking

2. **Optimization**
   - Fine-tune embeddings
   - Optimize chunking strategies
   - Implement caching
   - Add monitoring

3. **Production Features**
   - Authentication & authorization
   - Rate limiting
   - Error handling
   - Logging & observability

**Resources:**
- [LangChain Tutorials - RAG Applications](https://langchain-tutorials.com/lessons/rag-applications)
- [RAGAS Evaluation Framework](https://docs.ragas.io/)
- [Advanced RAG Techniques](https://www.pinecone.io/learn/advanced-rag-techniques/)

---

### Path 3: Advanced - Enterprise RAG Systems

**Duration:** 8-12 weeks

1. **Advanced Techniques**
   - Self-RAG implementation
   - Multi-agent RAG systems
   - Query expansion
   - Contextual compression

2. **Scalability**
   - Distributed vector databases
   - Load balancing
   - Microservices architecture
   - Kubernetes deployment

3. **Evaluation & Monitoring**
   - RAG evaluation metrics
   - A/B testing
   - Performance monitoring
   - User feedback loops

**Resources:**
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Pinecone Advanced RAG](https://www.pinecone.io/learn/advanced-rag-techniques/)
- [RAG Evaluation Guide](https://docs.ragas.io/)

---

## 🛠️ Hands-On Projects & Tutorials

### Project 1: Basic RAG System (Your Current System)
- **Tutorial**: [LangChain RAG Tutorial](https://python.langchain.com/docs/use_cases/question_answering/)
- **Skills**: Document loading, chunking, embeddings, retrieval, generation
- **Time**: 2-3 hours

### Project 2: Advanced RAG with Multiple Retrievers
- **Tutorial**: [Advanced RAG Techniques](https://www.pinecone.io/learn/advanced-rag-techniques/)
- **Skills**: MMR, hybrid search, re-ranking
- **Time**: 4-6 hours

### Project 3: RAG Evaluation System
- **Tutorial**: [RAGAS Documentation](https://docs.ragas.io/)
- **Skills**: Evaluation metrics, quality assessment
- **Time**: 3-4 hours

### Project 4: Production RAG API
- **Tutorial**: [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)
- **Skills**: Authentication, rate limiting, monitoring
- **Time**: 6-8 hours

---

## 📖 Additional Learning Resources

### Books
- **"LangChain in Action"** - Comprehensive guide to LangChain
- **"FastAPI Modern Python Web Development"** - FastAPI best practices
- **"Neural Information Retrieval"** - Deep dive into retrieval systems

### Online Courses
- **LangChain Academy**: https://academy.langchain.com/ (Free)
- **HuggingFace NLP Course**: https://huggingface.co/learn/nlp-course/ (Free)
- **FastAPI Course - freeCodeCamp**: https://www.youtube.com/watch?v=0sOvCWFmrtA (Free)

### Communities & Forums
- **LangChain Discord**: https://discord.gg/langchain
- **FastAPI GitHub Discussions**: https://github.com/tiangolo/fastapi/discussions
- **HuggingFace Forums**: https://discuss.huggingface.co/
- **Stack Overflow**: Tag `langchain`, `fastapi`, `rag`

### Blogs & Articles
- **Pinecone Learning Center**: https://www.pinecone.io/learn/
- **LangChain Blog**: https://blog.langchain.dev/
- **HuggingFace Blog**: https://huggingface.co/blog

### YouTube Channels
- **LangChain**: https://www.youtube.com/@LangChain
- **HuggingFace**: https://www.youtube.com/@HuggingFace
- **FastAPI**: https://www.youtube.com/@tiangolo

---

## 🔍 Evaluation & Testing Resources

### RAGAS (RAG Evaluation Framework)
- **Documentation**: https://docs.ragas.io/
- **GitHub**: https://github.com/explodinggradients/ragas
- **Tutorial**: https://docs.ragas.io/getting-started/quickstart

**Key Metrics:**
- Faithfulness
- Answer Relevance
- Context Precision
- Context Recall

### LangSmith (Observability)
- **Documentation**: https://docs.smith.langchain.com/
- **Platform**: https://smith.langchain.com/
- **Tutorial**: https://docs.smith.langchain.com/tracing

---

## 🚀 Next Steps for Your Learning

Based on your current implementation, here's what to learn next:

### Immediate (This Week)
1. ✅ **MMR Retrieval** - Improve retrieval diversity
   - Resource: [LangChain Retriever Types](https://python.langchain.com/docs/how_to/retrievers/)
   
2. ✅ **Better Prompting** - Enhance answer quality
   - Resource: [LangChain Prompt Engineering](https://python.langchain.com/docs/how_to/prompts/)

3. ✅ **Conversation Memory** - Add chat history
   - Resource: [LangChain Memory](https://python.langchain.com/docs/how_to/memory/)

### Short-term (This Month)
1. **Hybrid Search** - Combine semantic + keyword search
2. **Re-ranking** - Improve result quality
3. **Multiple File Types** - Support .txt, .md, .docx
4. **Evaluation Framework** - Add RAGAS for testing

### Long-term (Next 3 Months)
1. **Advanced RAG Techniques** - Self-RAG, query expansion
2. **Production Features** - Auth, rate limiting, monitoring
3. **Scalability** - Distributed systems, caching
4. **Fine-tuning** - Custom embeddings and models

---

## 📝 Practice Exercises

### Exercise 1: Improve Retrieval
- Implement MMR retrieval in `app/services/retriever.py`
- Compare results with similarity search
- Measure improvement with RAGAS

### Exercise 2: Enhance Prompts
- Create multiple prompt templates for different question types
- Add few-shot examples
- Implement prompt selection logic

### Exercise 3: Add Evaluation
- Set up RAGAS framework
- Create test dataset with Q&A pairs
- Measure system performance metrics

### Exercise 4: Support More Formats
- Add support for .txt files
- Add support for .md files
- Implement file type detection

---

## 🎯 Key Concepts to Master

### RAG Fundamentals
- **Retrieval**: How to find relevant documents
- **Augmentation**: How to enhance prompts with context
- **Generation**: How to generate accurate answers

### Vector Databases
- **Embeddings**: Converting text to vectors
- **Similarity Search**: Finding similar documents
- **Index Types**: Understanding FAISS indexes

### LLM Integration
- **Prompt Engineering**: Crafting effective prompts
- **Temperature**: Controlling randomness
- **Token Limits**: Managing context windows

### API Development
- **REST APIs**: Designing endpoints
- **Request Validation**: Using Pydantic
- **Error Handling**: Proper error responses

---

## 💡 Tips for Effective Learning

1. **Build Projects**: Apply concepts immediately
2. **Read Documentation**: Official docs are your best friend
3. **Join Communities**: Learn from others' experiences
4. **Experiment**: Try different approaches and compare
5. **Evaluate**: Measure improvements with metrics
6. **Iterate**: Continuously improve your system

---

## 📚 Documentation Quick Links

### Primary Resources
- [LangChain Docs](https://docs.langchain.com/)
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [FAISS Wiki](https://github.com/facebookresearch/faiss/wiki)
- [Sentence Transformers](https://www.sbert.net/)
- [Groq API Docs](https://console.groq.com/docs)

### Learning Platforms
- [LangChain Academy](https://academy.langchain.com/)
- [LangChain Tutorials](https://langchain-tutorials.com/)
- [HuggingFace Course](https://huggingface.co/learn/nlp-course/)

### Evaluation Tools
- [RAGAS](https://docs.ragas.io/)
- [LangSmith](https://docs.smith.langchain.com/)

---

**Happy Learning! 🚀**

*Last Updated: 2024*
*For questions or contributions, refer to the project README.md*
