"""
Wipe and reinitialize the database.

Deletes SQLite DB, ChromaDB data, and uploaded files, then
runs init_db.py to create fresh tables.

Usage:
    python reset_db.py
"""
import shutil
from pathlib import Path

from app.config import DATABASE_PATH, CHROMA_PATH, UPLOAD_PATH


def reset():
    # Wipe SQLite
    if DATABASE_PATH.exists():
        DATABASE_PATH.unlink()
        print(f"Deleted {DATABASE_PATH}")

    # Wipe ChromaDB
    if CHROMA_PATH.exists():
        shutil.rmtree(CHROMA_PATH)
        print(f"Deleted {CHROMA_PATH}")

    # Wipe uploaded files (keep .gitkeep)
    if UPLOAD_PATH.exists():
        for item in UPLOAD_PATH.iterdir():
            if item.name == '.gitkeep':
                continue
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
            print(f"Deleted {item}")

    # Reinitialize
    print()
    from init_db import init_database
    init_database()


if __name__ == '__main__':
    reset()
