# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

StudyAgent is an AI-powered study tool for art history courses that demonstrates a **hybrid architecture** — deterministic code paths where tasks are predictable, agentic LLM-driven processes where reasoning adds value. The core principle: "if the logic fits in a flowchart, it doesn't need an agent."

Key features:
- Upload lecture PDFs/documents → deterministic OCR and text extraction
- Chat with class materials → RAG agent with tool use (search + code execution)
- Generate flashcards → structured LLM outputs with term/definition pairs
- Export to Quizlet/Anki → deterministic CSV formatting

## Phase Status

**✅ Phase 1 - Foundation (COMPLETE)**
- SQLite database schema with all tables
- ChromaDB initialization
- File upload infrastructure with validation
- Flask + Tailwind CSS + HTMX interface (3-tab layout)
- Configuration management (.env, config.py)
- All ORM models created and tested
- Course files (run_agent.py, tools.py, usage.py) integrated

**✅ Phase 2 - Ingestion Pipeline (COMPLETE)**
- PDF text extraction (pypdf)
- Plain text and DOCX extraction
- Text chunking (800 tokens, 150 overlap)
- Embedding generation (OpenAI text-embedding-3-small)
- ChromaDB vector storage with metadata
- Integrated into Flask upload workflow

**✅ Phase 3 - RAG Chat Agent (COMPLETE)**
- ✅ Tool use implementation (search_class_materials + execute_python + search_web)
- ✅ Agent loop with OpenAI GPT-4o-mini (responses API with tool calling)
- ✅ Chat history persistence (SQLite database)
- ✅ Spelling correction with rapidfuzz (65% similarity threshold)
- ✅ Web search integration via Tavily API (optional, for historical context)
- ✅ Single-line chat input (Enter to send, standard chat UX)
- ⏸️ Usage tracking (deferred - usage.py needs OpenAI SDK update)

**✅ Phase 4 - Flashcard Generation (COMPLETE)**
- ✅ Flashcard generation agent (`app/agents/study_agent.py`)
  - RAG-based agent with tool use (search_class_materials + generate_flashcards_structured)
  - OpenAI Structured Outputs with Responses API
  - System prompt specialized for art history flashcards
  - Bugfix: Correct JSON parsing from response.output[0].content[0].text
- ✅ Export functionality (`app/pipelines/exporters.py`)
  - Quizlet TSV format (tab-separated)
  - Anki CSV format (proper escaping)
- ✅ Flask UI integration
  - Generate flashcards with topic input and count slider
  - Export to Quizlet/Anki formats
  - Database persistence (SQLite)
- ⏸️ Artwork image fetching (deferred to Phase 4.5)

**✅ Phase 5 - Flask Migration & UI Polish (COMPLETE)**
- Flask app factory with Blueprints (API + page routes)
- Tailwind CSS + HTMX frontend (SPA-like with partial rendering)
- Custom accessible dropdowns with keyboard navigation (class selector, set selector)
- Flashcard sets with auto-creation per generation
- Per-card inline editing (edit/delete individual flashcards)
- Custom migration system (`app/migrations/`)
- Dark mode with localStorage persistence
- Toast notifications and modal confirmations
- Chat markdown rendering (marked.js + DOMPurify)
- File management with cascade deletion (disk + SQLite + ChromaDB)

**✅ Phase 6 - RAG System Overhaul (COMPLETE)**
- Section-aware chunking (`app/pipelines/section_detector.py`)
  - Detects section headers (title-case, ALL-CAPS, markdown)
  - Strips parenthetical instructor notes before detection
  - Compound section names for subsections (e.g., `"Early Northern Renaissance > Terms, People, and Places to Know"`)
  - Prepends `[Section: ...]` prefix to chunk text for embedding disambiguation
- Upgraded search tool with three modes:
  - Semantic search (query), section filter (metadata), keyword filter (document text)
  - `list_sections` tool for discovering available sections
- Anti-hallucination system prompt (grounded in search results only)
- Source attribution (grouped by source/section at end of every response)
- Spelling correction hardened (common English word exclusion prevents "table" → "Marble")
- ChatMessage.to_dict() fix (removed `created_at` that broke OpenAI API on 2nd+ messages)
- Export serving from memory via BytesIO (no orphaned temp files)
- Flashcard limits increased to 60 (config, agent defaults, UI slider)
- Legacy `gradio_app.py` deleted
- `reset_db.py` script for clean DB wipes

## Architecture Philosophy

The codebase is split into two explicit layers:

**Deterministic Layer (`pipelines/`)**: Fixed code paths, no LLM reasoning
- File routing based on extensions
- OCR via Mathpix/Claude Vision
- Whisper transcription
- Text chunking and embedding
- CSV/TSV export

**Agentic Layer (`agents/`)**: LLM-directed processes where flexibility is needed
- RAG chat agent with tool use (search materials, execute Python)
- Study generation agent (flashcards/quizzes with optional reflection)

This architectural split is intentional and should be preserved — it demonstrates when agents add value vs. when deterministic code is faster/cheaper.

## Flask Frontend Architecture

The frontend uses Flask + Tailwind CSS + HTMX with vanilla JavaScript (no framework):

- **App Factory** (`app/__init__.py`): `create_app()` initializes SQLAlchemy, ChromaDB, runs migrations, registers blueprints
- **Blueprints**: `main_bp` (page routes, no prefix) + `api_bp` (REST API, `/api` prefix)
- **HTMX Partials**: Tab content loaded via `/partials/materials/<id>`, `/partials/chat/<id>`, `/partials/flashcards/<id>`
- **Template Hierarchy**: `base.html` (shell) → `index.html` (SPA page) → `partials/` (tab content, file lists, flashcard tables)
- **State Management**: `app.js` tracks `currentClassId`, `currentSetId`, `allFlashcards`, `flashcardPage` in module-level variables
- **Custom Dropdowns**: Built from scratch with ARIA roles, keyboard navigation (arrow keys, Enter, Escape), click-outside-to-close
- **Chat Rendering**: Assistant messages rendered as markdown (marked.js + DOMPurify), user messages as escaped plain text
- **Dark Mode**: CSS class toggle on `<html>` element, persisted via localStorage

## Project Structure

```
study_agent/
├── app/
│   ├── __init__.py              # App factory (create_app) ✅
│   ├── config.py                # API keys, DB paths, constants ✅
│   ├── extensions.py            # SQLAlchemy + ChromaDB init ✅
│   │
│   ├── models/                  # SQLAlchemy ORM models ✅
│   │   ├── __init__.py
│   │   ├── class_model.py       # Class container
│   │   ├── input_model.py       # Uploaded materials
│   │   ├── flashcard_model.py   # Generated flashcards (+ set_id FK)
│   │   ├── flashcard_set_model.py # Flashcard grouping by generation ✅
│   │   ├── quiz_model.py        # Generated quizzes
│   │   └── chat_model.py        # Conversation history
│   │
│   ├── migrations/              # Custom migration runner ✅
│   │   └── 001_flashcard_sets.py # Adds flashcard_sets table + set_id
│   │
│   ├── pipelines/               # ⚡ DETERMINISTIC LAYER
│   │   ├── __init__.py          ✅
│   │   ├── ingestion.py         # File routing + extraction ✅
│   │   ├── section_detector.py   # Document section detection ✅
│   │   ├── chunking.py          # Section-aware text splitting + embedding ✅
│   │   └── exporters.py         # Quizlet TSV / Anki CSV export ✅
│   │
│   ├── agents/                  # 🤖 AGENTIC LAYER
│   │   ├── __init__.py          ✅
│   │   ├── run_agent.py         # Agent execution loop (from course) ✅
│   │   ├── tools.py             # ToolBox class (from course) ✅
│   │   ├── chat_agent.py        # RAG chat + spelling correction ✅
│   │   └── study_agent.py       # Flashcard generation ✅
│   │
│   ├── utils/                   # Utility functions ✅
│   │   ├── __init__.py
│   │   ├── usage.py             # Cost tracking (deferred)
│   │   └── file_handler.py      # File upload handling ✅
│   │
│   ├── routes/                  # Flask Blueprints ✅
│   │   ├── __init__.py
│   │   ├── main.py              # Page routes + HTMX partials
│   │   └── api.py               # REST API (~22 endpoints)
│   │
│   ├── templates/               # Jinja2 templates ✅
│   │   ├── base.html            # Layout shell (CDNs, dark mode)
│   │   ├── index.html           # Main SPA page (tabs, dropdowns)
│   │   └── partials/
│   │       ├── materials_tab.html
│   │       ├── chat_tab.html
│   │       ├── flashcards_tab.html
│   │       ├── _file_list.html
│   │       ├── _flashcard_table.html
│   │       ├── _chat_messages.html
│   │       └── _modals.html     # 6 confirmation dialogs
│   │
│   └── static/                  # Frontend assets ✅
│       ├── css/custom.css       # Animations, markdown styles
│       └── js/app.js            # All client-side logic (~960 lines)
│
├── data/
│   ├── uploads/                 # User-uploaded files ✅
│   │   └── {class_id}/          # Organized by class
│   ├── app.db                   # SQLite metadata ✅
│   └── chroma/                  # ChromaDB vector store ✅
│
├── flask_app.py                 # Flask entry point ✅
├── init_db.py                   # Database initialization script ✅
├── reset_db.py                  # Wipe and reinitialize DB ✅
├── requirements.txt             # Python dependencies ✅
├── .env.example                 # Environment template ✅
├── .env                         # API keys (not committed) ✅
└── .gitignore                   # Git ignore rules ✅
```

## Tech Stack

| Component | Technology | Version | Notes |
|-----------|-----------|---------|-------|
| Backend | Flask + SQLAlchemy | Flask 3.0.0, SQLAlchemy 3.1.1 | App factory + Blueprints |
| Frontend | Tailwind CSS + HTMX | Tailwind CDN, HTMX 2.0.4 | SPA-like with partial rendering, no React |
| Markdown | marked.js + DOMPurify | Latest CDN | Chat message rendering |
| Metadata DB | SQLite | Built-in | Zero-config, single-file database |
| Vector DB | ChromaDB | ≥0.5.0 | One collection per class, NumPy 2.0 compatible |
| LLM APIs | OpenAI + Anthropic | openai 1.51.0, anthropic 0.40.0 | OpenAI for structured outputs, Claude for tool use |
| Math OCR | Mathpix API | Post-MVP | Handles equations, diagrams, STEM content |
| Audio | OpenAI Whisper | Post-MVP | Lecture recording transcription |
| Embeddings | text-embedding-3-small | — | Fast, cost-effective (1536 dimensions) |
| Python | Python 3.12 | 3.12+ | Virtual environment required |

### Compatibility Notes

- **ChromaDB 0.5.0+**: Required for NumPy 2.0 compatibility. Earlier versions fail on `np.float_` removal.
- **PyYAML ≥6.0.1**: Version 6.0 has build issues with Python 3.12. Use 6.0.1+ for pre-built wheels.

## Database Schema

### SQLite Tables

```sql
CREATE TABLE classes (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE inputs (
    id INTEGER PRIMARY KEY,
    class_id INTEGER NOT NULL REFERENCES classes(id),
    name TEXT NOT NULL,              -- user-given name
    input_type TEXT NOT NULL,        -- 'slides','recording','notes','textbook','quiz'
    file_path TEXT,
    raw_text TEXT,
    summary TEXT,                     -- auto-generated
    created_at TIMESTAMP
);

CREATE TABLE flashcard_sets (
    id INTEGER PRIMARY KEY,
    class_id INTEGER NOT NULL REFERENCES classes(id),
    name TEXT NOT NULL,              -- auto-named from topic (e.g., "Baroque Art")
    created_at TIMESTAMP
);

CREATE TABLE flashcards (
    id INTEGER PRIMARY KEY,
    class_id INTEGER NOT NULL REFERENCES classes(id),
    input_id INTEGER REFERENCES inputs(id),
    set_id INTEGER REFERENCES flashcard_sets(id),  -- groups cards by generation
    term TEXT NOT NULL,
    definition TEXT NOT NULL,
    image_url TEXT,                   -- public domain artwork image (optional)
    created_at TIMESTAMP
);

CREATE TABLE quizzes (
    id INTEGER PRIMARY KEY,
    class_id INTEGER NOT NULL REFERENCES classes(id),
    question TEXT NOT NULL,
    options TEXT,                     -- JSON array for MC
    answer TEXT NOT NULL,
    explanation TEXT,
    created_at TIMESTAMP
);

CREATE TABLE chat_messages (
    id INTEGER PRIMARY KEY,
    class_id INTEGER NOT NULL REFERENCES classes(id),
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    created_at TIMESTAMP
);
```

### Migrations

Custom migration runner (not Alembic) in `app/migrations/`. Migrations run automatically on app startup via `create_app()`. Each migration file has `up()` and `down()` functions operating on raw SQLite connections.

### ChromaDB Structure

- One collection per class: `class_{class_id}`
- Chunk metadata includes: `source` (input name), `chunk_idx`, `section` (detected section title)
- Section names may be compound for subsections: `"Parent > Subsection"`
- Chunk text is prefixed with `[Section: ...]` so the embedding itself captures section context
- Delete chunks by filtering: `collection.delete(where={"source": input_name})`

## Key Implementation Patterns

### 1. Deterministic Ingestion Pipeline
File routing based on extensions in `app/pipelines/ingestion.py`. Supports PDF (pypdf), DOCX (python-docx with heading style preservation), plain text, and PPTX. Section-aware chunking (800 chars, 150 overlap) in `app/pipelines/chunking.py` — detects document sections via `app/pipelines/section_detector.py`, prepends `[Section: ...]` prefix to each chunk, and stores section as ChromaDB metadata. Embedding via text-embedding-3-small.

### 2. RAG Chat Agent with Tool Use
Agent in `app/agents/chat_agent.py` has four tools:
- `search_class_materials(query, section, keyword)` — ChromaDB search with three modes: semantic (query), section filter (metadata), keyword filter (document text). Pass `""` for unused params.
- `list_sections()` — Returns all section names in the collection. Agent calls this first for comprehensive coverage.
- `execute_python` — Sandboxed code execution
- `search_web` — Tavily API (optional)

Uses OpenAI GPT-4o-mini via Responses API. Agent loop in `app/agents/run_agent.py` continues until the model produces a final text response. System prompt enforces anti-hallucination (only facts from search results) and source attribution (grouped "Sources:" section at end of every response).

### 3. Flashcard Generation with Structured Outputs
Agent in `app/agents/study_agent.py` uses `list_sections` + per-section `search_class_materials` to find relevant content, then generates flashcards via OpenAI Structured Outputs (JSON schema guarantees valid `{term, definition}` pairs). Auto-creates a `FlashcardSet` per generation, named after the topic. Default limit: 60 flashcards.

### 4. Code Execution Sandbox
Restricted `exec()` in `app/agents/chat_agent.py` with whitelisted modules (math, numpy, sympy, statistics, datetime). No builtins exposed. For production, consider Docker isolation.

## Domain Specialization: Art History

The system is tuned for art history content. See `app/agents/chat_agent.py` for the full system prompt.

- **Flashcard format**: Artist, Title, Date, Period, Medium, Style, Significance
- **Chunking strategy**: 800-token chunks with 150-token overlap, preserving artwork contexts
- **Spelling correction**: Fuzzy matching against terms from course materials (rapidfuzz, 65% threshold)

## Build Order

1. **Foundation**: SQLite schema, ChromaDB initialization, file uploads ✅
2. **Ingestion Pipeline**: PDF extraction, chunking, embedding ✅
3. **RAG Chat Agent**: Tool definitions, agent loop, chat interface ✅
4. **Flashcard Generation**: Structured outputs, CSV export ✅
5. **Flask Migration**: Gradio → Flask + Tailwind CSS + HTMX ✅
6. **UI Polish**: Flashcard sets, inline editing, custom dropdowns, dark mode ✅
7. **RAG Overhaul**: Section-aware chunking, upgraded search tools, anti-hallucination, source attribution ✅

Post-MVP: Quiz generation, reflection loop, artwork image fetching, multi-class dashboard, Whisper transcription, Mathpix OCR.

## Development Commands

**Setup (one-time):**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file from template
cp .env.example .env
# Edit .env and add your API keys

# Initialize database
python init_db.py
```

**Run the application:**
```bash
source venv/bin/activate  # Activate virtual environment
python flask_app.py       # Launch Flask UI at http://127.0.0.1:7860
```

**Reset database:**
```bash
python reset_db.py        # Wipe SQLite, ChromaDB, uploads and reinitialize
```

**Testing (Phase 2+):**
```bash
pytest  # Run tests
```

## API Keys Required

The following environment variables should be set in `.env`:

- `OPENAI_API_KEY`: For GPT models, Whisper, and embeddings
- `ANTHROPIC_API_KEY`: For Claude tool use agent
- `TAVILY_API_KEY`: For web search tool (optional, free tier at tavily.com)
- `MATHPIX_APP_ID` and `MATHPIX_APP_KEY`: For STEM OCR (post-MVP)

## Cost Considerations

The hybrid architecture minimizes costs:

| Operation | Agentic Cost | Deterministic Cost | Why Deterministic |
|-----------|-------------|-------------------|-------------------|
| Route PDF to OCR | ~$0.01-0.05 | ~$0.00 | File extension check is free |
| Chunk + embed 5 pages | ~$0.05+ | ~$0.001 | No reasoning needed |
| Transcribe 10-min lecture | ~$0.10+ | ~$0.06 | Direct API call is cheaper |

Agents are justified for:
- Chat with materials: Requires retrieval reasoning and synthesis (~$0.02-0.05/query)
- Flashcard generation with reflection: Quality improvement justifies cost (~$0.08-0.15 for 20 cards)

## Agentic Design Patterns Used

1. **Tool Use (ReAct)**: Chat agent decides whether to search, compute, or answer directly
2. **Code-as-Tool**: Math/data tasks executed in Python rather than reasoned about in text
3. **Planning** (post-MVP): "Help me study X" requires decomposition into subtasks
4. **Reflection** (post-MVP): Self-critique loop improves flashcard/quiz quality

The key insight: Everything else (file routing, OCR, transcription, chunking, embedding, CSV export, CRUD) is deliberately NOT agentic. This demonstrates understanding of when agents add value.

## Important Notes

- **Preserve the architectural split**: Keep `pipelines/` (deterministic) and `agents/` (agentic) separate. This is a core design principle, not just organization.
- **Art history focus for MVP**: Don't try to support multiple domains initially. The domain specialization in prompts and flashcard formats is intentional.
- **Flask is the frontend**: `flask_app.py` is the entry point.
- **Virtual environment required**: Always activate venv before running commands. Dependencies are isolated from system Python.

## Implementation Notes

Non-obvious decisions and gotchas not derivable from reading the code:

- **Spelling correction**: 65% similarity threshold (rapidfuzz `fuzz.ratio`). Common English words (200+ terms like "table", "format", "list") are excluded from correction to prevent "table" → "Marble" false matches. Only words not in `COMMON_ENGLISH_WORDS` set are candidates for correction.

- **Section detection**: `section_detector.py` strips parenthetical content before title-case detection (e.g., "Spanish Renaissance (Also called mannerism...)" → detects "Spanish Renaissance"). Subsection headers that repeat (e.g., "Terms, People, and Places to Know") get compound names: `"Parent > Subsection"`.

- **ChatMessage.to_dict()**: Only returns `role` and `content`. Including `created_at` caused `Unknown parameter: 'input[0].created_at'` errors with the OpenAI Responses API on 2nd+ messages.

- **Structured Outputs JSON parsing**: OpenAI's `response.output[0].content` is a list, not a string. Must extract via `response.output[0].content[0].text` before JSON parsing.

- **Flashcard sets**: Auto-created per generation, named after the topic. Migration `001_flashcard_sets.py` creates a "Previously generated" set per class to group pre-migration cards.

- **Web search**: Optional Tavily API integration. Works without API key (shows setup instructions). Add `TAVILY_API_KEY=tvly-xxx` to `.env`.

- **File deletion cascade**: Deleting a file removes data from 3 layers (disk, SQLite, ChromaDB). Chat history is intentionally preserved.

- **Export files**: Served from memory via `io.BytesIO` — no temp files written to disk, no orphaned exports.

- **Async in sync Flask**: Agent functions are async but Flask handlers are sync. Uses `asyncio.run()` as bridge.

### Future Enhancements

- **Response length limiting**: Agent can be verbose. Options: system prompt instruction (soft) or `max_tokens` parameter (hard). Start with system prompt.
- **Artwork image fetching**: Public domain images via Wikimedia/Met Museum APIs. Schema already has `image_url` column.
- **Quiz generation**: `quizzes` table exists but no agent implementation yet.
