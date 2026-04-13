"""
Text chunking and embedding generation for StudyAgent.

Section-aware text splitting with semantic chunking.
Embedding generation using OpenAI text-embedding-3-small.
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

from app.config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, OPENAI_API_KEY
from app.extensions import get_or_create_collection
from app.pipelines.section_detector import detect_sections


# Initialize OpenAI client
_openai_client = None

def get_openai_client():
    """Get or create OpenAI client singleton."""
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


def chunk_text(raw_text: str, chunk_size: int = None, overlap: int = None) -> list[dict]:
    """
    Split text into section-aware chunks using RecursiveCharacterTextSplitter.

    Detects document sections (headers, title-case lines, ALL-CAPS lines) and
    prepends section context to each chunk so embeddings capture section identity.

    Args:
        raw_text: Full text to chunk
        chunk_size: Max chars per chunk (default from config.CHUNK_SIZE)
        overlap: Char overlap between chunks (default from config.CHUNK_OVERLAP)

    Returns:
        List of dicts with keys:
        - "text": chunk text (with section prefix if detected)
        - "section": section title (empty string if no section)
    """
    chunk_size = chunk_size or CHUNK_SIZE
    overlap = overlap or CHUNK_OVERLAP

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    sections = detect_sections(raw_text)
    result = []

    for section_title, section_text in sections:
        chunks = text_splitter.split_text(section_text)
        for chunk in chunks:
            if section_title:
                prefixed_text = f"[Section: {section_title}]\n\n{chunk}"
            else:
                prefixed_text = chunk
            result.append({
                "text": prefixed_text,
                "section": section_title,
            })

    return result


def generate_embeddings(class_id: int, input_name: str, chunks: list[dict]) -> None:
    """
    Generate embeddings for chunks and store in ChromaDB.

    Uses OpenAI text-embedding-3-small for embedding generation.
    Stores in collection: class_{class_id}

    Args:
        class_id: Class ID for collection organization
        input_name: Source name for metadata
        chunks: List of chunk dicts with "text" and "section" keys

    Raises:
        Exception: If embedding generation or storage fails
    """
    if not chunks:
        raise ValueError("No chunks provided for embedding")

    # Get ChromaDB collection for this class
    collection = get_or_create_collection(class_id)

    # Extract text for embedding
    texts = [chunk["text"] for chunk in chunks]

    # Generate embeddings via OpenAI
    client = get_openai_client()
    response = client.embeddings.create(
        input=texts,
        model=EMBEDDING_MODEL
    )

    # Extract embeddings from response
    embeddings = [data.embedding for data in response.data]

    # Prepare metadata and IDs
    ids = [f"{input_name}_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "source": input_name,
            "chunk_idx": i,
            "section": chunk["section"],
        }
        for i, chunk in enumerate(chunks)
    ]

    # Store in ChromaDB
    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )


def delete_embeddings(class_id: int, input_name: str) -> None:
    """
    Delete all embeddings for a specific input.

    Used when user deletes an uploaded file.

    Args:
        class_id: Class ID
        input_name: Source name to delete
    """
    collection = get_or_create_collection(class_id)
    collection.delete(where={"source": input_name})
