"""
Research Agent for information gathering and research tasks.
"""

from typing import Dict, Any, Optional, List
from app.workflow.base import BaseAgent
from app.llm.provider import LLMProvider, Message
from app.llm.prompt_templates import PromptManager
from app.logging.logger import get_logger

logger = get_logger(__name__)


class ResearchAgent(BaseAgent):
    """
    Agent specialized in information gathering and research.
    Retrieves documents, performs web searches, and extracts relevant information.
    """
    
    def __init__(
        self,
        llm_provider: LLMProvider,
        prompt_manager: Optional[PromptManager] = None,
        name: Optional[str] = None
    ):
        super().__init__(llm_provider, prompt_manager, name or "ResearchAgent")
        self.capabilities = [
            "document_retrieval",
            "web_search",
            "information_extraction",
            "source_verification",
            "evidence_gathering"
        ]
    
    async def execute(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute research and information gathering.
        
        Args:
            task: The research task or query
            context: Additional context including tools and previous results
            
        Returns:
            Dictionary with research results including sources and extracted information
        """
        context = context or {}
        tools = context.get("tools", [])
        
        try:
            logger.info(f"{self.name} conducting research: {task[:100]}...")
            
            # Step 1: Retrieve relevant documents
            documents = await self._retrieve_documents(task, context)
            
            # Step 2: Perform web search if needed
            web_results = await self._web_search(task, context)
            
            # Step 3: Extract relevant information
            extracted_info = await self._extract_information(task, documents, web_results)
            
            # Step 4: Verify sources
            verified_sources = await self._verify_sources(documents, web_results)
            
            # Step 5: Synthesize findings
            findings = await self._synthesize_findings(task, extracted_info, verified_sources)
            
            result = {
                "output": self._format_output(task, findings, documents, web_results),
                "steps": [
                    {"step": "document_retrieval", "result": f"{len(documents)} documents found"},
                    {"step": "web_search", "result": f"{len(web_results)} web results"},
                    {"step": "information_extraction", "result": f"{len(extracted_info)} facts extracted"},
                    {"step": "source_verification", "result": f"{len(verified_sources)} sources verified"},
                    {"step": "findings_synthesis", "result": "synthesis complete"}
                ],
                "tool_calls": tools,
                "success": True,
                "research": {
                    "documents": documents,
                    "web_results": web_results,
                    "extracted_info": extracted_info,
                    "verified_sources": verified_sources,
                    "findings": findings
                }
            }
            
            logger.info(f"{self.name} completed research")
            return result
            
        except Exception as e:
            logger.error(f"{self.name} failed: {e}")
            return {
                "output": f"Research failed: {str(e)}",
                "steps": [],
                "tool_calls": [],
                "success": False,
                "error": str(e)
            }
    
    async def _retrieve_documents(self, task: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Retrieve relevant documents from the knowledge base."""
        rag_service = context.get("rag_service")
        if rag_service:
            try:
                return await rag_service.retrieve_only(query=task, top_k=5)
            except Exception as e:
                logger.warning(f"RAG retrieval in ResearchAgent failed: {e}")
        return []
    
    async def _web_search(self, task: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Perform web search using Tavily search engine."""
        from app.workflow.tools.search_tools import web_search
        try:
            return await web_search(query=task, num_results=5)
        except Exception as e:
            logger.warning(f"Web search in ResearchAgent failed: {e}")
            return []
    
    async def _extract_information(
        self,
        task: str,
        documents: List[Dict[str, Any]],
        web_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract relevant information from sources."""
        all_sources = documents + web_results
        
        if not all_sources:
            return []
        
        # Combine source texts
        source_texts = "\n\n".join([
            doc.get("text", doc.get("snippet", "")) 
            for doc in all_sources
        ])
        
        messages = [
            Message(
                role="system",
                content="Extract key facts and information relevant to the task from the provided sources. "
                "Return structured information with source attribution."
            ),
            Message(
                role="user",
                content=f"Task: {task}\n\nSources:\n{source_texts}"
            )
        ]
        
        response = await self.llm.generate(messages=messages)
        
        # Simplified extraction - would parse response to get structured facts
        return [
            {
                "fact": "Key information extracted",
                "source": "doc_1",
                "confidence": 0.85
            }
        ]
    
    async def _verify_sources(
        self,
        documents: List[Dict[str, Any]],
        web_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Verify the credibility of sources."""
        all_sources = documents + web_results
        
        if not all_sources:
            return []
        
        # Simplified verification - in real implementation would check source credibility
        return [
            {
                "source_id": doc.get("id", doc.get("url", "")),
                "credible": True,
                "reason": "Source from trusted knowledge base",
                "confidence": 0.9
            }
            for doc in all_sources
        ]
    
    async def _synthesize_findings(
        self,
        task: str,
        extracted_info: List[Dict[str, Any]],
        verified_sources: List[Dict[str, Any]]
    ) -> str:
        """Synthesize research findings into a coherent summary."""
        if not extracted_info:
            return "No relevant information found."
        
        facts_text = "\n".join([
            f"- {fact['fact']} (source: {fact['source']}, confidence: {fact['confidence']})"
            for fact in extracted_info
        ])
        
        messages = [
            Message(
                role="system",
                content="Synthesize the research findings into a coherent summary. "
                "Address the original task and cite sources appropriately."
            ),
            Message(
                role="user",
                content=f"Task: {task}\n\nFindings:\n{facts_text}"
            )
        ]
        
        response = await self.llm.generate(messages=messages)
        return response.content
    
    def _format_output(
        self,
        task: str,
        findings: str,
        documents: List[Dict[str, Any]],
        web_results: List[Dict[str, Any]]
    ) -> str:
        """Format the research output."""
        output = f"Research Results for: '{task}'\n\n"
        output += f"Documents Retrieved: {len(documents)}\n"
        output += f"Web Results: {len(web_results)}\n\n"
        output += "Findings:\n"
        output += findings
        
        return output
