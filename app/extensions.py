"""
Database extensions and singleton initialization.
SQLAlchemy for metadata, ChromaDB for vector embeddings.
"""
from flask_sqlalchemy import SQLAlchemy
import chromadb
from chromadb.config import Settings
from app.config import CHROMA_PATH, EMBEDDING_MODEL

# SQLAlchemy instance (initialized in Flask factory or Gradio app)
db = SQLAlchemy()

# ChromaDB client (singleton pattern)
_chroma_client = None

def get_chroma_client():
    """
    Get or create the ChromaDB persistent client.

    One collection per class: class_{class_id}
    Embedding function: OpenAI text-embedding-3-small (via default for MVP)
    """
    global _chroma_client

    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(
            path=str(CHROMA_PATH),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

    return _chroma_client

def get_or_create_collection(class_id: int):
    """
    Get or create a ChromaDB collection for a class.

    Collection naming: class_{class_id}
    Embedding: Manual embeddings (generated via OpenAI in chunking.py)

    Args:
        class_id: Integer class ID from SQLite

    Returns:
        chromadb.Collection
    """
    client = get_chroma_client()
    collection = client.get_or_create_collection(
        name=f"class_{class_id}",
        metadata={"class_id": class_id},
        embedding_function=None  # We provide embeddings manually
    )
    return collection

def delete_collection(class_id: int):
    """Delete a class's vector collection when class is deleted."""
    client = get_chroma_client()
    try:
        client.delete_collection(name=f"class_{class_id}")
    except ValueError:
        pass  # Collection doesn't exist, that's fine
