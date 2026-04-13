"""
Database initialization script.

Creates SQLite tables from ORM models and verifies ChromaDB connectivity.
Run this once after setting up .env with API keys.

Usage:
    python init_db.py
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from app.config import ensure_directories, validate_config, DATABASE_PATH
from app.extensions import db, get_chroma_client
from app.models import Class, Input, Flashcard, Quiz, ChatMessage
from flask import Flask

def init_database():
    """Create SQLite tables from ORM models."""
    print("Initializing StudyAgent database...")

    # Ensure directories exist
    ensure_directories()
    print(f"✓ Data directories created/verified")

    # Validate config
    try:
        validate_config()
        print(f"✓ Configuration valid (API keys loaded)")
    except ValueError as e:
        print(f"✗ Configuration error: {e}")
        sys.exit(1)

    # Create Flask app (minimal for SQLAlchemy)
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DATABASE_PATH}'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # Initialize SQLAlchemy
    db.init_app(app)

    # Create tables
    with app.app_context():
        db.create_all()
        print(f"✓ SQLite database created at {DATABASE_PATH}")

        # Verify tables
        from sqlalchemy import inspect
        inspector = inspect(db.engine)
        tables = inspector.get_table_names()
        print(f"  Tables: {', '.join(tables)}")

    # Initialize migration system
    from app.migrations import MigrationRunner
    migrations_dir = Path(__file__).parent / 'app' / 'migrations'
    runner = MigrationRunner(DATABASE_PATH, migrations_dir)

    # Create migration tracking table
    runner.ensure_migration_table()

    # Record baseline as version 0 (all tables already created above)
    import sqlite3
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR IGNORE INTO schema_migrations (version, description) VALUES (0, 'Baseline schema')"
    )
    conn.commit()
    conn.close()

    print(f"✓ Migration system initialized (baseline v0)")

    # Test ChromaDB connection
    try:
        client = get_chroma_client()
        print(f"✓ ChromaDB client initialized")

        # Create test collection and delete it
        test_coll = client.get_or_create_collection("test_init")
        client.delete_collection("test_init")
        print(f"✓ ChromaDB operations verified")
    except Exception as e:
        print(f"✗ ChromaDB error: {e}")
        sys.exit(1)

    print("\n" + "="*50)
    print("Database initialization complete!")
    print("="*50)
    print("\nNext steps:")
    print("1. Run: python flask_app.py")
    print("2. Upload a PDF to test ingestion pipeline")
    print("3. Chat with your materials!")

if __name__ == '__main__':
    init_database()
