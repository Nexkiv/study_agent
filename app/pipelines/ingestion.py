"""
Deterministic ingestion pipeline for StudyAgent.

File routing and text extraction based on file extensions.
No agent reasoning - pure Python code paths.
"""
from pathlib import Path
import pypdf
from docx import Document


def process_upload(file_path: str, class_id: int, input_name: str, input_type: str) -> str:
    """
    Main entry point for ingestion pipeline.
    Routes file to appropriate extractor based on extension.

    Args:
        file_path: Absolute path to uploaded file
        class_id: Class ID for organization
        input_name: User-provided name for this input
        input_type: Type of input (slides, textbook, notes, etc.)

    Returns:
        Extracted raw text as string

    Raises:
        ValueError: If file type is not supported
    """
    ext = Path(file_path).suffix.lower()

    # Route to appropriate extractor (deterministic)
    if ext in ('.txt', '.md'):
        return extract_plain_text(file_path)
    elif ext == '.docx':
        return extract_docx(file_path)
    elif ext == '.pdf':
        return extract_pdf(file_path)
    else:
        raise ValueError(
            f"Unsupported file type: {ext}. "
            f"Supported formats: .txt, .md, .docx, .pdf"
        )


def extract_plain_text(file_path: str) -> str:
    """
    Extract text from plain text files (.txt, .md).

    Args:
        file_path: Path to text file

    Returns:
        File contents as string
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def extract_docx(file_path: str) -> str:
    """
    Extract text from DOCX files using python-docx.

    Args:
        file_path: Path to .docx file

    Returns:
        Extracted text with paragraphs separated by newlines
    """
    doc = Document(file_path)
    paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
    return '\n\n'.join(paragraphs)


def extract_pdf(file_path: str) -> str:
    """
    Extract text from PDF files using pypdf.

    Args:
        file_path: Path to .pdf file

    Returns:
        Extracted text from all pages

    Note:
        For MVP, uses pypdf's basic text extraction.
        Post-MVP: Add Mathpix OCR for STEM content,
        Claude Vision for image-heavy PDFs.
    """
    text_parts = []

    with open(file_path, 'rb') as f:
        pdf_reader = pypdf.PdfReader(f)

        for page_num, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                if page_text.strip():
                    text_parts.append(page_text)
            except Exception as e:
                # Log but don't crash on individual page failures
                print(f"Warning: Failed to extract page {page_num + 1}: {e}")
                continue

    if not text_parts:
        raise ValueError("No text could be extracted from PDF")

    return '\n\n'.join(text_parts)
