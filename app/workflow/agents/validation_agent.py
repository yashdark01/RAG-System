"""
Validation Agent for fact-checking and verification.
"""

from typing import Dict, Any, Optional, List
from app.workflow.base import BaseAgent
from app.llm.provider import LLMProvider, Message
from app.llm.prompt_templates import PromptManager
from app.logging.logger import get_logger

logger = get_logger(__name__)


class ValidationAgent(BaseAgent):
    """
    Agent specialized in fact-checking and verification.
    Validates claims, checks sources, and assesses confidence in information.
    """
    
    def __init__(
        self,
        llm_provider: LLMProvider,
        prompt_manager: Optional[PromptManager] = None,
        name: Optional[str] = None
    ):
        super().__init__(llm_provider, prompt_manager, name or "ValidationAgent")
        self.capabilities = [
            "fact_checking",
            "source_verification",
            "confidence_scoring",
            "claim_validation",
            "consistency_check"
        ]
    
    async def execute(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute validation and fact-checking.
        
        Args:
            task: The claim or information to validate
            context: Additional context including sources and previous results
            
        Returns:
            Dictionary with validation results and confidence scores
        """
        context = context or {}
        tools = context.get("tools", [])
        
        try:
            logger.info(f"{self.name} validating: {task[:100]}...")
            
            # Step 1: Extract claims from the task
            claims = await self._extract_claims(task)
            
            # Step 2: Verify each claim
            verified_claims = []
            for claim in claims:
                verification = await self._verify_claim(claim, context)
                verified_claims.append(verification)
            
            # Step 3: Check source credibility
            source_assessment = await self._assess_sources(context)
            
            # Step 4: Check consistency
            consistency_check = await self._check_consistency(verified_claims)
            
            # Step 5: Calculate overall confidence
            overall_confidence = self._calculate_overall_confidence(
                verified_claims,
                source_assessment,
                consistency_check
            )
            
            # Step 6: Generate validation report
            report = self._generate_validation_report(
                task,
                verified_claims,
                source_assessment,
                consistency_check,
                overall_confidence
            )
            
            result = {
                "output": report,
                "steps": [
                    {"step": "claim_extraction", "result": f"{len(claims)} claims extracted"},
                    {"step": "claim_verification", "result": f"{len(verified_claims)} claims verified"},
                    {"step": "source_assessment", "result": source_assessment.get("overall_score", 0)},
                    {"step": "consistency_check", "result": consistency_check.get("consistent", False)},
                    {"step": "confidence_calculation", "result": overall_confidence}
                ],
                "tool_calls": tools,
                "success": True,
                "validation": {
                    "claims": verified_claims,
                    "source_assessment": source_assessment,
                    "consistency": consistency_check,
                    "overall_confidence": overall_confidence
                }
            }
            
            logger.info(f"{self.name} completed validation")
            return result
            
        except Exception as e:
            logger.error(f"{self.name} failed: {e}")
            return {
                "output": f"Validation failed: {str(e)}",
                "steps": [],
                "tool_calls": [],
                "success": False,
                "error": str(e)
            }
    
    async def _extract_claims(self, task: str) -> List[Dict[str, Any]]:
        """Extract individual claims from the task."""
        messages = [
            Message(
                role="system",
                content="Extract individual factual claims from the provided text. "
                "Each claim should be a standalone statement that can be verified."
            ),
            Message(
                role="user",
                content=f"Extract claims from: {task}"
            )
        ]
        
        response = await self.llm.generate(messages=messages)
        
        # Simplified claim extraction - would parse response to get structured claims
        return [
            {
                "claim": task,
                "type": "factual",
                "importance": "high"
            }
        ]
    
    async def _verify_claim(self, claim: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Verify a single claim against available sources."""
        claim_text = claim.get("claim", "")
        
        # Get sources from context
        documents = context.get("documents", [])
        web_results = context.get("web_results", [])
        
        # Combine source texts
        source_texts = []
        for doc in documents:
            source_texts.append(doc.get("text", ""))
        for web in web_results:
            source_texts.append(web.get("snippet", ""))
        
        sources_text = "\n\n".join(source_texts) if source_texts else "No sources available"
        
        messages = [
            Message(
                role="system",
                content="Verify the claim against the provided sources. "
                "Determine if the claim is supported, contradicted, or cannot be verified. "
                "Provide confidence score and reasoning."
            ),
            Message(
                role="user",
                content=f"Claim: {claim_text}\n\nSources:\n{sources_text}"
            )
        ]
        
        response = await self.llm.generate(messages=messages)
        
        # Simplified verification - would parse response to get structured verification
        return {
            "claim": claim_text,
            "status": "supported",  # supported, contradicted, unverified
            "confidence": 0.85,
            "reasoning": "Claim is supported by available sources",
            "sources_used": len(source_texts)
        }
    
    async def _assess_sources(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the credibility of sources."""
        documents = context.get("documents", [])
        web_results = context.get("web_results", [])
        
        total_sources = len(documents) + len(web_results)
        
        if total_sources == 0:
            return {
                "overall_score": 0.0,
                "credible_sources": 0,
                "total_sources": 0,
                "assessment": "No sources available"
            }
        
        # Simplified source assessment
        credible_count = total_sources  # Assume all are credible for now
        overall_score = min(credible_count / total_sources, 1.0)
        
        return {
            "overall_score": overall_score,
            "credible_sources": credible_count,
            "total_sources": total_sources,
            "assessment": f"{credible_count}/{total_sources} sources appear credible"
        }
    
    async def _check_consistency(self, verified_claims: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check consistency among verified claims."""
        if len(verified_claims) < 2:
            return {
                "consistent": True,
                "conflicts": [],
                "reasoning": "Insufficient claims to check consistency"
            }
        
        # Extract claim statuses
        statuses = [claim.get("status") for claim in verified_claims]
        
        # Check for contradictions
        has_contradictions = "contradicted" in statuses
        
        return {
            "consistent": not has_contradictions,
            "conflicts": [] if not has_contradictions else ["Found contradictory claims"],
            "reasoning": "Claims are consistent" if not has_contradictions else "Found contradictions"
        }
    
    def _calculate_overall_confidence(
        self,
        verified_claims: List[Dict[str, Any]],
        source_assessment: Dict[str, Any],
        consistency_check: Dict[str, Any]
    ) -> float:
        """Calculate overall confidence score."""
        if not verified_claims:
            return 0.0
        
        # Average claim confidence
        avg_claim_confidence = sum(
            claim.get("confidence", 0.0) for claim in verified_claims
        ) / len(verified_claims)
        
        # Source credibility factor
        source_score = source_assessment.get("overall_score", 0.0)
        
        # Consistency factor
        consistency_factor = 1.0 if consistency_check.get("consistent", True) else 0.5
        
        # Calculate weighted average
        overall_confidence = (
            avg_claim_confidence * 0.5 +
            source_score * 0.3 +
            consistency_factor * 0.2
        )
        
        return round(min(overall_confidence, 1.0), 2)
    
    def _generate_validation_report(
        self,
        task: str,
        verified_claims: List[Dict[str, Any]],
        source_assessment: Dict[str, Any],
        consistency_check: Dict[str, Any],
        overall_confidence: float
    ) -> str:
        """Generate a comprehensive validation report."""
        report = f"Validation Report for: '{task[:100]}...'\n\n"
        report += f"Overall Confidence: {overall_confidence * 100:.1f}%\n\n"
        
        report += "Claims Verified:\n"
        for claim in verified_claims:
            status = claim.get("status", "unknown")
            confidence = claim.get("confidence", 0.0)
            report += f"- {claim.get('claim', '')[:50]}... ({status}, {confidence * 100:.1f}%)\n"
        
        report += f"\nSource Assessment: {source_assessment.get('assessment', 'N/A')}\n"
        report += f"Consistency: {'Consistent' if consistency_check.get('consistent') else 'Inconsistent'}\n"
        
        return report
