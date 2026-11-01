"""
Synthesis Agent for combining information and generating answers.
"""

from typing import Dict, Any, Optional, List
from app.workflow.base import BaseAgent
from app.llm.provider import LLMProvider, Message
from app.llm.prompt_templates import PromptManager
from app.logging.logger import get_logger

logger = get_logger(__name__)


class SynthesisAgent(BaseAgent):
    """
    Agent specialized in synthesizing information and generating comprehensive answers.
    Combines research findings, extracts key insights, and generates coherent responses.
    """
    
    def __init__(
        self,
        llm_provider: LLMProvider,
        prompt_manager: Optional[PromptManager] = None,
        name: Optional[str] = None
    ):
        super().__init__(llm_provider, prompt_manager, name or "SynthesisAgent")
        self.capabilities = [
            "text_synthesis",
            "summarization",
            "answer_generation",
            "insight_extraction",
            "response_formatting"
        ]
    
    async def execute(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute information synthesis and answer generation.
        
        Args:
            task: The synthesis task or question to answer
            context: Additional context including research results and previous agent outputs
            
        Returns:
            Dictionary with synthesized answer and supporting information
        """
        context = context or {}
        tools = context.get("tools", [])
        
        try:
            logger.info(f"{self.name} synthesizing: {task[:100]}...")
            
            # Step 1: Gather information from context
            research_results = context.get("research_results", {})
            query_analysis = context.get("query_analysis", {})
            
            # Step 2: Extract key insights
            insights = await self._extract_insights(task, research_results, query_analysis)
            
            # Step 3: Generate comprehensive answer
            answer = await self._generate_answer(task, insights, research_results)
            
            # Step 4: Create summary
            summary = await self._create_summary(answer)
            
            # Step 5: Format response
            formatted_response = await self._format_response(task, answer, summary, insights)
            
            result = {
                "output": formatted_response,
                "steps": [
                    {"step": "insight_extraction", "result": f"{len(insights)} insights extracted"},
                    {"step": "answer_generation", "result": "answer generated"},
                    {"step": "summary_creation", "result": "summary created"},
                    {"step": "response_formatting", "result": "response formatted"}
                ],
                "tool_calls": tools,
                "success": True,
                "synthesis": {
                    "insights": insights,
                    "answer": answer,
                    "summary": summary,
                    "confidence": self._calculate_confidence(insights, research_results)
                }
            }
            
            logger.info(f"{self.name} completed synthesis")
            return result
            
        except Exception as e:
            logger.error(f"{self.name} failed: {e}")
            return {
                "output": f"Synthesis failed: {str(e)}",
                "steps": [],
                "tool_calls": [],
                "success": False,
                "error": str(e)
            }
    
    async def _extract_insights(
        self,
        task: str,
        research_results: Dict[str, Any],
        query_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract key insights from research results."""
        documents = research_results.get("documents", [])
        web_results = research_results.get("web_results", [])
        extracted_info = research_results.get("extracted_info", [])
        
        # Combine all available information
        all_info = []
        
        for doc in documents:
            all_info.append(doc.get("text", ""))
        
        for web in web_results:
            all_info.append(web.get("snippet", ""))
        
        for info in extracted_info:
            all_info.append(info.get("fact", ""))
        
        if not all_info:
            return []
        
        combined_text = "\n\n".join(all_info)
        
        messages = [
            Message(
                role="system",
                content="Extract key insights from the provided information. "
                "Focus on the most important and relevant points that address the task."
            ),
            Message(
                role="user",
                content=f"Task: {task}\n\nInformation:\n{combined_text}"
            )
        ]
        
        response = await self.llm.generate(messages=messages)
        
        # Simplified insight extraction - would parse response to get structured insights
        return [
            {
                "insight": "Key insight extracted from information",
                "importance": "high",
                "source_count": len(all_info)
            }
        ]
    
    async def _generate_answer(
        self,
        task: str,
        insights: List[Dict[str, Any]],
        research_results: Dict[str, Any]
    ) -> str:
        """Generate a comprehensive answer based on insights."""
        if not insights:
            return "Unable to generate answer due to insufficient information."
        
        insights_text = "\n".join([
            f"- {insight['insight']} (importance: {insight['importance']})"
            for insight in insights
        ])
        
        messages = [
            Message(
                role="system",
                content="Generate a comprehensive and accurate answer to the task based on the insights. "
                "Be thorough, well-structured, and cite sources when appropriate."
            ),
            Message(
                role="user",
                content=f"Task: {task}\n\nInsights:\n{insights_text}"
            )
        ]
        
        response = await self.llm.generate(messages=messages)
        return response.content
    
    async def _create_summary(self, answer: str) -> str:
        """Create a concise summary of the answer."""
        if not answer:
            return ""
        
        messages = [
            Message(
                role="system",
                content="Create a concise summary (2-3 sentences) of the provided answer."
            ),
            Message(
                role="user",
                content=f"Answer:\n{answer}"
            )
        ]
        
        response = await self.llm.generate(messages=messages)
        return response.content
    
    async def _format_response(
        self,
        task: str,
        answer: str,
        summary: str,
        insights: List[Dict[str, Any]]
    ) -> str:
        """Format the final response."""
        formatted = f"Answer to: '{task}'\n\n"
        formatted += f"Summary: {summary}\n\n"
        formatted += "Detailed Answer:\n"
        formatted += answer
        formatted += "\n\nKey Insights:\n"
        
        for insight in insights:
            formatted += f"- {insight['insight']}\n"
        
        return formatted
    
    def _calculate_confidence(
        self,
        insights: List[Dict[str, Any]],
        research_results: Dict[str, Any]
    ) -> float:
        """Calculate confidence score for the synthesis."""
        if not insights:
            return 0.0
        
        # Base confidence from number of insights
        base_confidence = min(len(insights) * 0.2, 0.8)
        
        # Boost from research results
        document_count = len(research_results.get("documents", []))
        web_count = len(research_results.get("web_results", []))
        source_boost = min((document_count + web_count) * 0.05, 0.2)
        
        total_confidence = min(base_confidence + source_boost, 1.0)
        return round(total_confidence, 2)
