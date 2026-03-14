"""
File upload handling with validation and secure storage.

Saves files to data/uploads/{class_id}/{sanitized_filename}
Validates file types against whitelist.
"""
import os
import uuid
from pathlib import Path
from werkzeug.utils import secure_filename
from app.config import UPLOAD_PATH

# Allowed file extensions for MVP
ALLOWED_EXTENSIONS = {
    'pdf', 'txt', 'md', 'docx',
}

def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_upload(file, class_id: int, original_filename: str) -> str:
    """
    Save uploaded file to data/uploads/{class_id}/{filename}.

    Args:
        file: File object (from Gradio or Flask request)
        class_id: Class ID for subdirectory organization
        original_filename: Original filename from user

    Returns:
        Relative file path from UPLOAD_PATH (for database storage)

    Raises:
        ValueError: If file type not allowed
    """
    if not allowed_file(original_filename):
        raise ValueError(
            f"File type not allowed. Supported: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # Secure filename
    safe_name = secure_filename(original_filename)

    # Add unique prefix to avoid collisions
    unique_name = f"{uuid.uuid4().hex[:8]}_{safe_name}"

    # Class subdirectory
    class_dir = UPLOAD_PATH / str(class_id)
    class_dir.mkdir(parents=True, exist_ok=True)

    # Full path
    file_path = class_dir / unique_name

    # Save file (handle both Gradio and Flask file objects)
    if hasattr(file, 'save'):
        # Flask FileStorage object
        file.save(str(file_path))
    else:
        # Gradio file or path string
        import shutil
        if isinstance(file, str):
            shutil.copy(file, file_path)
        else:
            with open(file_path, 'wb') as f:
                f.write(file.read())

    # Return relative path for database
    relative_path = f"{class_id}/{unique_name}"
    return relative_path

def get_upload_path(relative_path: str) -> Path:
    """Convert relative path from database to absolute path."""
    return UPLOAD_PATH / relative_path

def delete_upload(relative_path: str):
    """Delete an uploaded file from disk."""
    file_path = get_upload_path(relative_path)
    if file_path.exists():
        file_path.unlink()
