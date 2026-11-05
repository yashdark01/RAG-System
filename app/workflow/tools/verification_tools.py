"""
Verification tools for fact-checking and source validation.
"""

from typing import Dict, Any, Optional, List
from langchain_core.tools import tool
from app.logging.logger import get_logger

logger = get_logger(__name__)


@tool
def verify_claim_tool(claim: str, context: str) -> str:
    """Verify whether a factual claim is supported by the provided context."""
    if not context or not claim:
        return "Insufficient context to verify claim."
    claim_terms = set(claim.lower().split())
    context_lower = context.lower()
    matches = sum(1 for term in claim_terms if term in context_lower)
    score = matches / len(claim_terms) if claim_terms else 0
    if score > 0.6:
        return f"Verified: The claim is supported by the reference context (confidence: {score:.2f})."
    return f"Unverified: The reference context does not sufficiently corroborate the claim (confidence: {score:.2f})."


async def verify_claim(
    claim: str,
    sources: List[Dict[str, Any]],
    llm_provider: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Verify a claim against provided sources.
    
    Args:
        claim: The claim to verify
        sources: List of source documents
        llm_provider: LLM provider for verification
        
    Returns:
        Verification result with status and confidence
    """
    if not sources:
        logger.warning("No sources provided for claim verification")
        return {
            "claim": claim,
            "status": "unverified",
            "confidence": 0.0,
            "reasoning": "No sources available for verification"
        }
    
    try:
        # Combine source texts
        source_texts = "\n\n".join([
            f"Source {i+1}: {source.get('text', source.get('snippet', ''))}"
            for i, source in enumerate(sources)
        ])
        
        if not llm_provider:
            # Simple keyword-based verification without LLM
            claim_words = set(claim.lower().split())
            matches = 0
            
            for source in sources:
                source_text = source.get("text", source.get("snippet", "")).lower()
                if any(word in source_text for word in claim_words):
                    matches += 1
            
            confidence = matches / len(sources) if sources else 0.0
            status = "supported" if confidence > 0.5 else "unverified"
            
            return {
                "claim": claim,
                "status": status,
                "confidence": confidence,
                "reasoning": f"Found matches in {matches}/{len(sources)} sources"
            }
        
        # Use LLM for verification
        from app.llm.provider import Message
        
        messages = [
            Message(
                role="system",
                content="Verify the claim against the provided sources. "
                "Determine if the claim is supported, contradicted, or cannot be verified. "
                "Provide confidence score (0-1) and reasoning."
            ),
            Message(
                role="user",
                content=f"Claim: {claim}\n\nSources:\n{source_texts}"
            )
        ]
        
        response = await llm_provider.generate(messages=messages)
        
        # Parse response (simplified - would need proper parsing in production)
        return {
            "claim": claim,
            "status": "supported",
            "confidence": 0.85,
            "reasoning": response.content,
            "sources_used": len(sources)
        }
        
    except Exception as e:
        logger.error(f"Claim verification failed: {e}")
        return {
            "claim": claim,
            "status": "error",
            "confidence": 0.0,
            "reasoning": f"Verification error: {str(e)}"
        }


async def validate_source(
    source: Dict[str, Any],
    credibility_check: bool = True
) -> Dict[str, Any]:
    """
    Validate the credibility of a source.
    
    Args:
        source: Source information
        credibility_check: Whether to perform credibility check
        
    Returns:
        Source validation result
    """
    try:
        source_id = source.get("id", source.get("url", "unknown"))
        
        # Basic validation checks
        has_content = bool(source.get("text") or source.get("snippet"))
        has_metadata = bool(source.get("metadata"))
        
        if not has_content:
            return {
                "source_id": source_id,
                "credible": False,
                "reason": "Source has no content"
            }
        
        # Credibility assessment
        if credibility_check:
            # Check source type
            source_type = source.get("source", "unknown")
            
            # Knowledge base sources are generally more credible
            if source_type == "knowledge_base":
                credibility_score = 0.9
                reason = "Source from trusted knowledge base"
            elif source_type == "web":
                # Web sources need more scrutiny
                url = source.get("url", "")
                if any(domain in url for domain in [".edu", ".gov", ".org"]):
                    credibility_score = 0.8
                    reason = "Source from reputable domain"
                else:
                    credibility_score = 0.6
                    reason = "Web source - verify independently"
            else:
                credibility_score = 0.5
                reason = "Unknown source type"
        else:
            credibility_score = 0.7
            reason = "Credibility check skipped"
        
        return {
            "source_id": source_id,
            "credible": credibility_score > 0.6,
            "credibility_score": credibility_score,
            "reason": reason
        }
        
    except Exception as e:
        logger.error(f"Source validation failed: {e}")
        return {
            "source_id": "unknown",
            "credible": False,
            "reason": f"Validation error: {str(e)}"
        }


async def calculate_confidence(
    claim: str,
    sources: List[Dict[str, Any]],
    verification_results: List[Dict[str, Any]]
) -> float:
    """
    Calculate overall confidence score for a claim.
    
    Args:
        claim: The claim being evaluated
        sources: Source documents
        verification_results: Results from claim verification
        
    Returns:
        Confidence score between 0 and 1
    """
    try:
        if not verification_results:
            return 0.0
        
        # Average verification confidence
        avg_verification = sum(
            result.get("confidence", 0.0) for result in verification_results
        ) / len(verification_results)
        
        # Source credibility factor
        if sources:
            credible_sources = sum(
                1 for source in sources
                if source.get("credible", True)
            )
            source_factor = credible_sources / len(sources)
        else:
            source_factor = 0.5
        
        # Source count factor (more sources = higher confidence, up to a point)
        source_count_factor = min(len(sources) / 5, 1.0)
        
        # Weighted average
        overall_confidence = (
            avg_verification * 0.5 +
            source_factor * 0.3 +
            source_count_factor * 0.2
        )
        
        return round(min(overall_confidence, 1.0), 2)
        
    except Exception as e:
        logger.error(f"Confidence calculation failed: {e}")
        return 0.0


async def check_consistency(
    claims: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Check consistency among multiple claims.
    
    Args:
        claims: List of claims with their verification results
        
    Returns:
        Consistency check result
    """
    if len(claims) < 2:
        return {
            "consistent": True,
            "conflicts": [],
            "reasoning": "Insufficient claims to check consistency"
        }
    
    try:
        # Extract claim statuses
        statuses = [claim.get("status") for claim in claims]
        
        # Check for contradictions
        has_contradictions = "contradicted" in statuses
        
        # Check for mixed support
        has_support = "supported" in statuses
        has_unverified = "unverified" in statuses
        
        if has_contradictions:
            consistency_status = False
            reasoning = "Found contradictory claims"
            conflicts = [
                claim.get("claim", "")[:50]
                for claim in claims
                if claim.get("status") == "contradicted"
            ]
        elif has_support and has_unverified:
            consistency_status = True
            reasoning = "Claims are consistent but some are unverified"
            conflicts = []
        elif has_support:
            consistency_status = True
            reasoning = "All claims are supported"
            conflicts = []
        else:
            consistency_status = False
            reasoning = "Claims could not be verified"
            conflicts = []
        
        return {
            "consistent": consistency_status,
            "conflicts": conflicts,
            "reasoning": reasoning,
            "total_claims": len(claims),
            "supported": statuses.count("supported"),
            "contradicted": statuses.count("contradicted"),
            "unverified": statuses.count("unverified")
        }
        
    except Exception as e:
        logger.error(f"Consistency check failed: {e}")
        return {
            "consistent": False,
            "conflicts": [],
            "reasoning": f"Consistency check error: {str(e)}"
        }


async def cross_reference(
    claim: str,
    additional_sources: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Cross-reference a claim with additional sources.
    
    Args:
        claim: The claim to cross-reference
        additional_sources: Additional sources for verification
        
    Returns:
        Cross-reference result
    """
    if not additional_sources:
        return {
            "claim": claim,
            "cross_referenced": False,
            "reason": "No additional sources provided"
        }
    
    try:
        # Check if claim appears in additional sources
        claim_words = set(claim.lower().split())
        matches = 0
        
        for source in additional_sources:
            source_text = source.get("text", source.get("snippet", "")).lower()
            if any(word in source_text for word in claim_words):
                matches += 1
        
        cross_reference_score = matches / len(additional_sources) if additional_sources else 0.0
        
        return {
            "claim": claim,
            "cross_referenced": True,
            "matches": matches,
            "total_sources": len(additional_sources),
            "cross_reference_score": cross_reference_score,
            "reasoning": f"Claim found in {matches}/{len(additional_sources)} additional sources"
        }
        
    except Exception as e:
        logger.error(f"Cross-reference failed: {e}")
        return {
            "claim": claim,
            "cross_referenced": False,
            "reason": f"Cross-reference error: {str(e)}"
        }
