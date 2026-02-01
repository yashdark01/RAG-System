# app/core/llm.py
from langchain_google_genai import ChatGoogleGenerativeAI
import os


def get_llm(temperature=0.7, model_name="gemini-1.5-flash-latest"):
    """
    Initialize and return a Google Gemini language model.
    Gemini has a generous free tier that's perfect for development.
    
    Available models (use -latest suffix):
    - gemini-1.5-flash-latest (fastest, recommended)
    - gemini-1.5-pro-latest (more capable)
    - gemini-2.0-flash-exp (experimental)

    Args:
        temperature (float): The temperature setting for the model (0.0 to 1.0).
        model_name (str): The name of the Gemini model to use.
    Returns:
        ChatGoogleGenerativeAI: An instance of the Gemini model.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found in environment variables. "
            "Get your free API key from https://aistudio.google.com/app/apikey"
        )

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        google_api_key=api_key,
        convert_system_message_to_human=True,
        max_output_tokens=2048,
    )
    return llm