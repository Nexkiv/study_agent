"""
StudyAgent - AI-powered study tool for art history.

This package contains the core application logic organized into:
- models/: SQLAlchemy ORM models (SQLite)
- pipelines/: Deterministic processing (OCR, chunking, embedding)
- agents/: Agentic processing (RAG chat, study generation)
- routes/: Flask Blueprints
"""
import os
from flask import Flask
from app.config import (
    SQLALCHEMY_DATABASE_URI,
    SQLALCHEMY_TRACK_MODIFICATIONS,
    PROJECT_ROOT,
    DATABASE_PATH,
    ensure_directories,
    validate_config,
)
from app.extensions import db, get_chroma_client

__version__ = '0.1.0'


def create_app():
    """Flask application factory."""
    validate_config()
    ensure_directories()

    app = Flask(
        __name__,
        template_folder=str(PROJECT_ROOT / 'app' / 'templates'),
        static_folder=str(PROJECT_ROOT / 'app' / 'static'),
    )
    app.config['SQLALCHEMY_DATABASE_URI'] = SQLALCHEMY_DATABASE_URI
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = SQLALCHEMY_TRACK_MODIFICATIONS
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', os.urandom(24))
    app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB upload limit

    db.init_app(app)
    get_chroma_client()

    # Run database migrations
    from app.migrations import MigrationRunner
    migrations_dir = PROJECT_ROOT / 'app' / 'migrations'
    runner = MigrationRunner(DATABASE_PATH, migrations_dir)
    try:
        runner.run_migrations()
    except Exception as e:
        print(f"Warning: Migration failed: {e}")

    # Register blueprints
    from app.routes.main import main_bp
    from app.routes.api import api_bp
    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp, url_prefix='/api')

    return app
