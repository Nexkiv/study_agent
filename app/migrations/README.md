# Database Migrations

This directory contains database schema migrations.

## Creating a New Migration

1. Create a new file: `NNN_descriptive_name.py` (e.g., `002_add_user_preferences.py`)
2. Increment version number from the last migration
3. Implement the `Migration` subclass:

```python
from app.migrations import Migration
import sqlite3

class MyMigration(Migration):
    version = 2  # Increment from last migration
    description = "What this migration does"

    def up(self, conn: sqlite3.Connection):
        # Apply changes
        cursor = conn.cursor()
        cursor.execute("ALTER TABLE ...")

    def down(self, conn: sqlite3.Connection):
        # Optional: revert changes
        pass
```

## Migration Naming Convention

- Use format: `NNN_description.py`
- NNN = zero-padded version number (001, 002, etc.)
- description = snake_case description

Examples:
- `001_add_extraction_method.py`
- `002_add_user_preferences.py`
- `003_add_quiz_answers.py`

## How Migrations Work

1. On app startup, `MigrationRunner` scans this directory
2. Compares discovered migrations to `schema_migrations` table
3. Applies any pending migrations in version order
4. Records applied migrations to prevent re-running

## Manual Migration

If needed, run migrations manually:

```bash
python -c "from app.migrations import MigrationRunner; from app.config import DATABASE_PATH, PROJECT_ROOT; runner = MigrationRunner(DATABASE_PATH, PROJECT_ROOT / 'app/migrations'); runner.run_migrations()"
```
