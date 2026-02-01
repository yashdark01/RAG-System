# app/services/loader.py
from langchain_community.document_loaders import PyPDFLoader, TextLoader


def load_pdf(file_path: str):
    """
    Load a PDF document from the specified file path.

    Args:
        file_path (str): The path to the PDF file.
    Returns:
        List[Document]: A list of Document objects containing the text from the PDF.
    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file cannot be loaded.
    """
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        return documents
    except FileNotFoundError:
        raise FileNotFoundError(f"PDF file not found: {file_path}")
    except Exception as e:
        raise ValueError(f"Error loading PDF: {str(e)}")


def load_text(file_path: str):
    """
    Load a text document from the specified file path.

    Args:
        file_path (str): The path to the text file.
    Returns:
        List[Document]: A list of Document objects containing the text from the file.
    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file cannot be loaded.
    """
    try:
        loader = TextLoader(file_path)
        documents = loader.load()
        return documents
    except FileNotFoundError:
        raise FileNotFoundError(f"Text file not found: {file_path}")
    except Exception as e:
        raise ValueError(f"Error loading text file: {str(e)}")