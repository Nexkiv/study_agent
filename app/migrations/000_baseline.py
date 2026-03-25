"""
Migration 000: Baseline schema.

This is a no-op migration that records the initial database schema.
All tables are created by db.create_all() in init_db.py, but we record
this as migration 0 so future migrations start from version 1.

Tables included in baseline:
- classes: Course containers
- inputs: Uploaded files with extraction_method field
- flashcards: Generated flashcards
- quizzes: Generated quizzes (future)
- chat_messages: Conversation history
"""
import sqlite3
from app.migrations import Migration


class Baseline(Migration):
    version = 0
    description = "Baseline schema (all tables created by init_db.py)"

    def up(self, conn: sqlite3.Connection):
        """
        No-op: tables already created by db.create_all().
        This migration just records that the baseline schema is in place.
        """
        pass

    def down(self, conn: sqlite3.Connection):
        """No-op: this is the baseline, no rollback."""
        pass
