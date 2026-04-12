"""
StudyAgent Flask Application Entry Point.

Run:
    python flask_app.py

Replaces gradio_app.py with Flask + Tailwind CSS + HTMX.
"""
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

from app import create_app

app = create_app()

if __name__ == '__main__':
    import os
    # Only print on the main process, not the reloader child
    if not os.environ.get('WERKZEUG_RUN_MAIN'):
        print("✓ StudyAgent starting at http://127.0.0.1:7860")
    app.run(host='127.0.0.1', port=7860, debug=True)
