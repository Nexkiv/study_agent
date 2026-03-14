"""
Configuration management for StudyAgent.
Loads environment variables and provides typed config access.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# API Keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
MATHPIX_APP_ID = os.getenv('MATHPIX_APP_ID')
MATHPIX_APP_KEY = os.getenv('MATHPIX_APP_KEY')

# Database paths (absolute)
DATABASE_PATH = PROJECT_ROOT / os.getenv('DATABASE_PATH', 'data/app.db')
CHROMA_PATH = PROJECT_ROOT / os.getenv('CHROMA_PATH', 'data/chroma')
UPLOAD_PATH = PROJECT_ROOT / os.getenv('UPLOAD_PATH', 'data/uploads')

# SQLAlchemy connection string
SQLALCHEMY_DATABASE_URI = f'sqlite:///{DATABASE_PATH}'
SQLALCHEMY_TRACK_MODIFICATIONS = False

# Embeddings config
EMBEDDING_MODEL = 'text-embedding-3-small'
EMBEDDING_DIMENSIONS = 1536

# Chunking config
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# Model defaults
DEFAULT_CHAT_MODEL = 'claude-sonnet-4-20250514'
DEFAULT_STRUCTURED_MODEL = 'gpt-5-mini'

def ensure_directories():
    """Create data directories if they don't exist."""
    UPLOAD_PATH.mkdir(parents=True, exist_ok=True)
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

def validate_config():
    """Raise error if required API keys are missing."""
    missing = []
    if not OPENAI_API_KEY:
        missing.append('OPENAI_API_KEY')
    if not ANTHROPIC_API_KEY:
        missing.append('ANTHROPIC_API_KEY')

    if missing:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            "Copy .env.example to .env and add your API keys."
        )
