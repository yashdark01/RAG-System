"""
Query Agent for understanding and decomposing user queries.
"""

from typing import Dict, Any, Optional, List
from app.workflow.base import BaseAgent
from app.llm.provider import LLMProvider, Message
from app.llm.prompt_templates import PromptManager
from app.logging.logger import get_logger

logger = get_logger(__name__)


class QueryAgent(BaseAgent):
    """
    Agent specialized in query understanding and decomposition.
    Analyzes user queries to extract intent, entities, and decompose complex queries.
    """
    
    def __init__(
        self,
        llm_provider: LLMProvider,
        prompt_manager: Optional[PromptManager] = None,
        name: Optional[str] = None
    ):
        super().__init__(llm_provider, prompt_manager, name or "QueryAgent")
        self.capabilities = [
            "query_analysis",
            "query_decomposition",
            "intent_detection",
            "entity_extraction",
            "query_clarification"
        ]
    
    async def execute(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute query analysis and decomposition.
        
        Args:
            task: The user query to analyze
            context: Additional context including tools and previous results
            
        Returns:
            Dictionary with analysis results including intent, entities, and sub-queries
        """
        context = context or {}
        tools = context.get("tools", [])
        
        try:
            logger.info(f"{self.name} analyzing query: {task[:100]}...")
            
            # Step 1: Analyze query intent
            intent_analysis = await self._analyze_intent(task)
            
            # Step 2: Extract entities
            entities = await self._extract_entities(task)
            
            # Step 3: Decompose complex queries if needed
            sub_queries = await self._decompose_query(task, intent_analysis)
            
            # Step 4: Clarify ambiguous queries if needed
            clarifications = await self._check_clarification_needed(task, intent_analysis)
            
            result = {
                "output": self._format_output(
                    task,
                    intent_analysis,
                    entities,
                    sub_queries,
                    clarifications
                ),
                "steps": [
                    {"step": "intent_analysis", "result": intent_analysis},
                    {"step": "entity_extraction", "result": entities},
                    {"step": "query_decomposition", "result": sub_queries},
                    {"step": "clarification_check", "result": clarifications}
                ],
                "tool_calls": [],
                "success": True,
                "analysis": {
                    "intent": intent_analysis,
                    "entities": entities,
                    "sub_queries": sub_queries,
                    "clarifications": clarifications
                }
            }
            
            logger.info(f"{self.name} completed query analysis")
            return result
            
        except Exception as e:
            logger.error(f"{self.name} failed: {e}")
            return {
                "output": f"Query analysis failed: {str(e)}",
                "steps": [],
                "tool_calls": [],
                "success": False,
                "error": str(e)
            }
    
    async def _analyze_intent(self, query: str) -> Dict[str, Any]:
        """Analyze the intent of the user query."""
        messages = [
            Message(
                role="system",
                content="You are a query analysis expert. Analyze the user's query and determine its intent. "
                "Classify the intent as: information_search, comparison, explanation, how_to, definition, or other. "
                "Provide confidence score and reasoning."
            ),
            Message(
                role="user",
                content=f"Analyze this query: {query}"
            )
        ]
        
        response = await self.llm.generate(messages=messages)
        
        # Parse the response to extract structured intent
        return {
            "primary_intent": "information_search",  # Simplified, would parse from response
            "confidence": 0.85,
            "reasoning": "Query seeks specific information",
            "complexity": "medium"
        }
    
    async def _extract_entities(self, query: str) -> List[Dict[str, Any]]:
        """Extract key entities from the query."""
        messages = [
            Message(
                role="system",
                content="Extract key entities (people, places, concepts, dates, etc.) from the query."
            ),
            Message(
                role="user",
                content=f"Extract entities from: {query}"
            )
        ]
        
        response = await self.llm.generate(messages=messages)
        
        # Simplified entity extraction
        return [
            {"type": "concept", "value": "machine learning", "confidence": 0.9}
        ]
    
    async def _decompose_query(self, query: str, intent: Dict[str, Any]) -> List[str]:
        """Decompose complex queries into sub-queries."""
        if intent.get("complexity") == "simple":
            return [query]
        
        messages = [
            Message(
                role="system",
                content="Break down complex queries into simpler sub-queries that can be answered independently."
            ),
            Message(
                role="user",
                content=f"Decompose this query: {query}"
            )
        ]
        
        response = await self.llm.generate(messages=messages)
        
        # Simplified decomposition
        return [query]  # Would parse response to get actual sub-queries
    
    async def _check_clarification_needed(self, query: str, intent: Dict[str, Any]) -> List[str]:
        """Check if query needs clarification."""
        messages = [
            Message(
                role="system",
                content="Determine if the query is ambiguous and needs clarification. "
                "List specific questions that would help clarify the query."
            ),
            Message(
                role="user",
                content=f"Check if clarification needed: {query}"
            )
        ]
        
        response = await self.llm.generate(messages=messages)
        
        # Simplified check
        return []  # Would parse response to get actual clarification questions
    
    def _format_output(
        self,
        query: str,
        intent: Dict[str, Any],
        entities: List[Dict[str, Any]],
        sub_queries: List[str],
        clarifications: List[str]
    ) -> str:
        """Format the analysis output."""
        output = f"Query Analysis for: '{query}'\n\n"
        output += f"Intent: {intent.get('primary_intent')} (confidence: {intent.get('confidence')})\n"
        output += f"Entities Found: {len(entities)}\n"
        output += f"Sub-queries: {len(sub_queries)}\n"
        
        if clarifications:
            output += f"Clarifications Needed: {len(clarifications)}\n"
        
        return output
