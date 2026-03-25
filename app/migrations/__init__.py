"""
Simple migration system for StudyAgent.
Automatically applies database schema changes on app startup.

Usage:
    from app.migrations import MigrationRunner
    runner = MigrationRunner(db_path, migrations_dir)
    runner.run_migrations()
"""
import sqlite3
from pathlib import Path
from typing import List
import importlib.util


class Migration:
    """Base class for all migrations."""
    version: int
    description: str

    def up(self, conn: sqlite3.Connection):
        """Apply migration (must be implemented by subclasses)."""
        raise NotImplementedError

    def down(self, conn: sqlite3.Connection):
        """Rollback migration (optional)."""
        pass


class MigrationRunner:
    """Discovers and runs database migrations."""

    def __init__(self, db_path: Path, migrations_dir: Path):
        self.db_path = db_path
        self.migrations_dir = migrations_dir

    def ensure_migration_table(self):
        """Create schema_migrations table if not exists."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                description TEXT NOT NULL,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

    def get_applied_migrations(self) -> set:
        """Get set of already-applied migration versions."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT version FROM schema_migrations")
        applied = {row[0] for row in cursor.fetchall()}
        conn.close()
        return applied

    def discover_migrations(self) -> List[Migration]:
        """Load all migration files from migrations/ directory."""
        migrations = []

        for file_path in sorted(self.migrations_dir.glob("*.py")):
            if file_path.name.startswith("__"):
                continue

            # Import migration module
            spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
            if spec is None or spec.loader is None:
                continue

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find Migration subclass
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and
                    issubclass(attr, Migration) and
                    attr is not Migration):
                    migrations.append(attr())

        return sorted(migrations, key=lambda m: m.version)

    def run_migrations(self):
        """Apply all pending migrations."""
        self.ensure_migration_table()
        applied = self.get_applied_migrations()
        pending_migrations = [
            m for m in self.discover_migrations()
            if m.version not in applied
        ]

        if not pending_migrations:
            return

        conn = sqlite3.connect(self.db_path)

        for migration in pending_migrations:
            print(f"Applying migration {migration.version}: {migration.description}")

            try:
                migration.up(conn)

                # Record migration
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO schema_migrations (version, description) VALUES (?, ?)",
                    (migration.version, migration.description)
                )
                conn.commit()

                print(f"✓ Migration {migration.version} applied successfully")
            except Exception as e:
                conn.rollback()
                print(f"✗ Migration {migration.version} failed: {e}")
                raise

        conn.close()
