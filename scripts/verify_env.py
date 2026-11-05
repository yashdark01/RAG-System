#!/usr/bin/env python3
"""
Pre-flight environment and dependency verification script.
Checks system requirements, environment variables, directories, and model backends.
"""

import sys
import os
from pathlib import Path

def print_status(component: str, ok: bool, details: str = ""):
    symbol = "✅" if ok else "❌"
    msg = f"{symbol} [{component}]: {details}" if details else f"{symbol} [{component}]"
    print(msg)
    return ok

def main():
    print("=" * 60)
    print(" 🛠️  RAG System Pre-Flight Environment Diagnostic")
    print("=" * 60)

    all_ok = True

    # 1. Python version check
    py_ver = sys.version_info
    py_ok = (py_ver.major == 3 and py_ver.minor >= 10)
    all_ok &= print_status(
        "Python Version", 
        py_ok, 
        f"{py_ver.major}.{py_ver.minor}.{py_ver.micro} (>= 3.10 required)"
    )

    # 2. Check configuration and paths
    try:
        from app.config import settings
        print_status("App Configuration", True, f"Loaded '{settings.APP_NAME}' v{settings.APP_VERSION}")
    except Exception as e:
        all_ok &= print_status("App Configuration", False, str(e))
        return 1

    # 3. Storage directories
    for dir_name in ["data", "data/vector_store", "data/uploads", "logs"]:
        path = Path(dir_name)
        path.mkdir(parents=True, exist_ok=True)
        exists = path.is_dir() and os.access(path, os.W_OK)
        all_ok &= print_status(f"Directory '{dir_name}'", exists, "Writable")

    # 4. LLM API Key status
    keys_found = []
    if settings.GEMINI_API_KEY or settings.GOOGLE_API_KEY:
        keys_found.append("Google/Gemini")
    if settings.GROQ_API_KEY:
        keys_found.append("Groq")
    if settings.OPENAI_API_KEY:
        keys_found.append("OpenAI")
    if settings.TAVILY_API_KEY:
        keys_found.append("Tavily (Web Search)")

    if keys_found:
        print_status("API Keys Detected", True, ", ".join(keys_found))
    else:
        print_status("API Keys Detected", False, "No active LLM or Search API keys found in .env (offline fallback will be used)")

    # 5. Core dependencies check
    deps = [
        ("FastAPI", "fastapi"),
        ("LangChain Core", "langchain_core"),
        ("LangChain Community", "langchain_community"),
        ("FAISS", "faiss"),
        ("SQLAlchemy", "sqlalchemy"),
        ("Redis Client", "redis"),
    ]
    for name, module in deps:
        try:
            __import__(module)
            print_status(f"Dependency {name}", True, "Installed")
        except ImportError as e:
            all_ok &= print_status(f"Dependency {name}", False, str(e))

    print("=" * 60)
    if all_ok:
        print("🎉 Environment is healthy and ready for development and deployment!")
        return 0
    else:
        print("⚠️  Some warnings or errors were identified above. Check logs/details.")
        return 0

if __name__ == "__main__":
    sys.exit(main())
