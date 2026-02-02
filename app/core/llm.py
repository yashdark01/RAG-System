import logging
import os
from typing import Optional

from langchain_groq import ChatGroq

logger = logging.getLogger(__name__)

# Default Groq model – using llama-3.1-8b-instant (fast and reliable)
# Alternative models: mixtral-8x7b-32768, llama-3.3-70b-versatile
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"


def get_llm(temperature: float = 0.7, model_name: Optional[str] = None) -> ChatGroq:
    """
    Initialize and return a Groq-hosted LLM.

    This replaces the previous Google Gemini setup to avoid 404 model errors.

    Requirements:
    - Install `langchain-groq` (see requirements.txt)
    - Set GROQ_API_KEY in your environment / .env file.
      You can get a free key from: https://console.groq.com/keys

    Args:
        temperature: Sampling temperature for the model.
        model_name: Groq model name to use. If not provided,
            defaults to DEFAULT_GROQ_MODEL.

    Returns:
        ChatGroq: Configured Groq chat model instance.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        error_msg = (
            "GROQ_API_KEY not found in environment variables. "
            "Get a free API key from https://console.groq.com/keys "
            "and add it to your .env file."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    if model_name is None:
        model_name = DEFAULT_GROQ_MODEL

    logger.info("Initializing Groq LLM with model: %s", model_name)
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name=model_name,
        temperature=temperature,
    )
    logger.info("Groq LLM initialized successfully")
    return llm
