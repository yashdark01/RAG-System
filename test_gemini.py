import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("ERROR: GOOGLE_API_KEY not found in environment")
    exit(1)

print(f"API Key found (first 10 chars): {api_key[:10]}...")
print("\n" + "="*60)
print("Step 1: Listing available models from Google API")
print("="*60 + "\n")

# Try to list models using the Google Generative AI API
try:
    # Use v1 API to list models
    url = "https://generativelanguage.googleapis.com/v1/models"
    headers = {"x-goog-api-key": api_key}
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        models_data = response.json()
        if "models" in models_data:
            print("Available models:")
            print("-" * 60)
            for model in models_data["models"]:
                name = model.get("name", "Unknown")
                display_name = model.get("displayName", "")
                supported_methods = model.get("supportedGenerationMethods", [])
                if "generateContent" in supported_methods:
                    print(f"✓ {name}")
                    if display_name:
                        print(f"  Display Name: {display_name}")
                    print(f"  Methods: {', '.join(supported_methods)}")
                    print()
        else:
            print("No models found in response")
            print(f"Response: {models_data}")
    else:
        print(f"Failed to list models. Status: {response.status_code}")
        print(f"Response: {response.text}")
        
except Exception as e:
    print(f"Error listing models: {e}")

print("\n" + "="*60)
print("Step 2: Testing models with LangChain")
print("="*60 + "\n")

from langchain_google_genai import ChatGoogleGenerativeAI

models_to_test = [
    "gemini-1.5-flash",  # Try without models/ prefix first
    "gemini-1.5-pro",
    "gemini-pro",
    "models/gemini-1.5-flash",
    "models/gemini-1.5-pro",
    "models/gemini-pro",
]

for model_name in models_to_test:
    try:
        print(f"Testing: {model_name}...", end=" ")
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
        )
        result = llm.invoke("Say hello")
        print(f"✓ SUCCESS!")
        print(f"  Response: {result.content[:50]}...")
        print(f"  → Use this model name in your code: '{model_name}'\n")
        break
    except Exception as e:
        error_str = str(e)
        if "404" in error_str or "NOT_FOUND" in error_str:
            print("✗ 404 Not Found")
        else:
            print(f"✗ Error: {error_str[:100]}")