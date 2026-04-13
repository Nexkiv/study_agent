"""
Deterministic ingestion pipeline for StudyAgent.

File routing and text extraction based on file extensions.
No agent reasoning - pure Python code paths.
"""
from pathlib import Path
from typing import Tuple
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
    elif ext == '.pptx':
        from app.pipelines.ocr import extract_pptx
        return extract_pptx(file_path)
    else:
        raise ValueError(
            f"Unsupported file type: {ext}. "
            f"Supported formats: .txt, .md, .docx, .pdf, .pptx"
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
    parts = []
    for para in doc.paragraphs:
        if not para.text.strip():
            continue
        # Preserve heading styles as markdown headers for section detection
        if para.style and para.style.name and para.style.name.startswith('Heading'):
            level_str = para.style.name.replace('Heading', '').strip()
            try:
                level = int(level_str)
            except ValueError:
                level = 1
            parts.append(f"{'#' * level} {para.text}")
        else:
            parts.append(para.text)
    return '\n\n'.join(parts)


def extract_pdf(file_path: str) -> Tuple[str, int, bool]:
    """
    Extract text from PDF files using pypdf.

    Args:
        file_path: Path to .pdf file

    Returns:
        Tuple of (extracted_text, page_count, needs_ocr_flag)

    Note:
        For MVP, uses pypdf's basic text extraction.
        If extraction quality is poor, suggests OCR via needs_ocr flag.
        Phase 2.5: OCR fallback with Tesseract/Mathpix/Claude Vision.
    """
    from app.pipelines.ocr import detect_poor_extraction

    text_parts = []
    page_count = 0

    with open(file_path, 'rb') as f:
        pdf_reader = pypdf.PdfReader(f)
        page_count = len(pdf_reader.pages)

        for page_num, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                if page_text.strip():
                    text_parts.append(page_text)
            except Exception as e:
                # Log but don't crash on individual page failures
                print(f"Warning: Failed to extract page {page_num + 1}: {e}")
                continue

    full_text = '\n\n'.join(text_parts) if text_parts else ""

    # Check if OCR might help
    needs_ocr, reason = detect_poor_extraction(full_text, page_count)

    if needs_ocr:
        print(f"OCR suggested: {reason}")

    # Still raise error if completely empty
    if not full_text.strip():
        raise ValueError("No text could be extracted from PDF (try OCR)")

    return full_text, page_count, needs_ocr
