"""
Migration 001: Add flashcard_sets table and link flashcards to sets.

Groups existing flashcards into a "Previously generated" set per class.
"""
import sqlite3
from app.migrations import Migration


class AddFlashcardSets(Migration):
    version = 1
    description = "Add flashcard_sets table and link flashcards to sets"

    def up(self, conn: sqlite3.Connection):
        # Create flashcard_sets table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS flashcard_sets (
                id INTEGER PRIMARY KEY,
                class_id INTEGER NOT NULL REFERENCES classes(id),
                name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Add set_id column to flashcards (nullable for backward compat)
        try:
            conn.execute(
                "ALTER TABLE flashcards ADD COLUMN set_id INTEGER REFERENCES flashcard_sets(id)"
            )
        except sqlite3.OperationalError:
            pass  # Column already exists

        # Migrate existing flashcards: group by class_id into a "Previously generated" set
        cursor = conn.execute("SELECT DISTINCT class_id FROM flashcards WHERE set_id IS NULL")
        for (class_id,) in cursor.fetchall():
            conn.execute(
                "INSERT INTO flashcard_sets (class_id, name) VALUES (?, ?)",
                (class_id, "Previously generated")
            )
            set_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            conn.execute(
                "UPDATE flashcards SET set_id = ? WHERE class_id = ? AND set_id IS NULL",
                (set_id, class_id)
            )

        conn.commit()

    def down(self, conn: sqlite3.Connection):
        conn.execute("DROP TABLE IF EXISTS flashcard_sets")
        # SQLite doesn't support DROP COLUMN, so set_id stays
