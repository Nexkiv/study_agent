# StudyAgent

AI-powered study tool that turns your course materials into interactive study aids. Upload lecture PDFs, chat with your materials via RAG, and generate flashcards exportable to Quizlet and Anki. Built with a hybrid architecture: deterministic pipelines where tasks are predictable, LLM-driven processes where reasoning adds value.

## Features

- **Upload course materials** — PDF, DOCX, PPTX, and plain text. Section-aware chunking with vector embeddings.
- **Chat with your materials** — RAG-powered agent with semantic search, keyword search, section filtering, and web search. Anti-hallucination guardrails with source attribution.
- **Generate flashcards** — LLM intent parsing understands what you want (terms, people, places, artworks, etc.). Parallel search + structured output generation with fuzzy deduplication and category filtering.
- **Export** — Quizlet TSV and Anki CSV formats, served from memory.
- **Dark mode** — Persisted via localStorage.

## Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY (required) and optionally TAVILY_API_KEY

# Initialize database
python init_db.py
```

## Usage

```bash
source venv/bin/activate
python flask_app.py
```

Open http://127.0.0.1:7860 in your browser.

1. **Create a class** using the dropdown in the top bar.
2. **Upload materials** in the Materials tab — drag and drop or click to upload PDFs, DOCX, PPTX, or text files. Files are chunked and embedded automatically.
3. **Chat** in the Chat tab — ask questions about your materials. The agent searches your uploads and cites sources.
4. **Generate flashcards** in the Flashcards tab — enter a topic (e.g., "Key Terms", "Artists and People", "Places to Know") and click Generate. Flashcards are grouped into sets.
5. **Edit flashcards** inline — click any card to edit its term or definition, or delete individual cards.
6. **Export** — click "Export to Quizlet" or "Export to Anki" to download your flashcards.

## Tech Stack

- **Backend**: Flask + SQLAlchemy + SQLite
- **Frontend**: Tailwind CSS + HTMX + vanilla JS
- **Vector DB**: ChromaDB (one collection per class)
- **LLM**: OpenAI GPT-4o-mini (chat, flashcards, structured outputs)
- **Embeddings**: text-embedding-3-small
- **Search**: Semantic, keyword, and section-based via ChromaDB

## Reset

To wipe all data and start fresh:

```bash
python reset_db.py
```
