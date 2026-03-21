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
- Gradio MVP interface (3-panel layout)
- Configuration management (.env, config.py)
- All ORM models created and tested
- Course files (run_agent.py, tools.py, usage.py) integrated

**✅ Phase 2 - Ingestion Pipeline (COMPLETE)**
- PDF text extraction (pypdf)
- Plain text and DOCX extraction
- Text chunking (800 tokens, 150 overlap)
- Embedding generation (OpenAI text-embedding-3-small)
- ChromaDB vector storage with metadata
- Integrated into Gradio upload workflow

**🚧 Phase 3 - RAG Chat Agent (TODO)**
- Tool use implementation (search + code execution)
- Agent loop with Anthropic Claude
- Chat history persistence
- Usage tracking (re-enable usage.py)

**🚧 Phase 4 - Flashcard Generation (TODO)**
- OpenAI Structured Outputs
- CSV export for Quizlet/Anki
- Artwork image fetching (post-MVP)

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

## Project Structure

```
study_agent/
├── app/
│   ├── __init__.py              # Package initialization ✅
│   ├── config.py                # API keys, DB paths, constants ✅
│   ├── extensions.py            # SQLAlchemy + ChromaDB init ✅
│   │
│   ├── models/                  # SQLAlchemy ORM models ✅
│   │   ├── __init__.py
│   │   ├── class_model.py       # Class container
│   │   ├── input_model.py       # Uploaded materials
│   │   ├── flashcard_model.py   # Generated flashcards
│   │   ├── quiz_model.py        # Generated quizzes
│   │   └── chat_model.py        # Conversation history
│   │
│   ├── pipelines/               # ⚡ DETERMINISTIC LAYER
│   │   ├── __init__.py          ✅
│   │   ├── ingestion.py         # File routing + extraction ✅
│   │   ├── chunking.py          # Text splitting + embedding ✅
│   │   └── exporters.py         # Phase 4: CSV/TSV export
│   │
│   ├── agents/                  # 🤖 AGENTIC LAYER
│   │   ├── __init__.py          ✅
│   │   ├── run_agent.py         # Agent execution loop (from course) ✅
│   │   ├── tools.py             # ToolBox class (from course) ✅
│   │   ├── chat_agent.py        # Phase 3: RAG + tool use
│   │   └── study_agent.py       # Phase 4: Flashcard generation
│   │
│   ├── utils/                   # Utility functions ✅
│   │   ├── __init__.py
│   │   ├── usage.py             # Cost tracking (deferred to Phase 3)
│   │   └── file_handler.py      # File upload handling ✅
│   │
│   ├── routes/                  # Flask Blueprints (post-MVP)
│   ├── templates/               # Jinja2 templates (post-MVP)
│   └── static/                  # CSS/JS assets (post-MVP)
│
├── data/
│   ├── uploads/                 # User-uploaded files ✅
│   │   └── {class_id}/          # Organized by class
│   ├── app.db                   # SQLite metadata ✅
│   └── chroma/                  # ChromaDB vector store ✅
│
├── init_db.py                   # Database initialization script ✅
├── gradio_app.py                # Gradio MVP interface ✅
├── requirements.txt             # Python dependencies ✅
├── .env.example                 # Environment template ✅
├── .env                         # API keys (not committed) ✅
└── .gitignore                   # Git ignore rules ✅
```

**Legend:** ✅ = Implemented

## Tech Stack

| Component | Technology | Version | Notes |
|-----------|-----------|---------|-------|
| Backend | Flask + SQLAlchemy | Flask 3.0.0, SQLAlchemy 3.1.1 | Python-native, modular with Blueprints |
| Metadata DB | SQLite | Built-in | Zero-config, single-file database |
| Vector DB | ChromaDB | ≥0.5.0 | One collection per class, NumPy 2.0 compatible |
| LLM APIs | OpenAI + Anthropic | openai 1.51.0, anthropic 0.40.0 | OpenAI for structured outputs, Claude for tool use |
| Math OCR | Mathpix API | Post-MVP | Handles equations, diagrams, STEM content |
| Audio | OpenAI Whisper | Post-MVP | Lecture recording transcription |
| Embeddings | text-embedding-3-small | Phase 2 | Fast, cost-effective |
| Frontend (MVP) | Gradio Blocks | ≥5.0.0 | Rapid prototyping, compatible with modern huggingface_hub |
| Frontend (v2) | Tailwind CSS + HTMX | Post-MVP | Production UI, no React needed |
| Python | Python 3.12 | 3.12+ | Virtual environment required |

### Compatibility Notes

- **ChromaDB 0.5.0+**: Required for NumPy 2.0 compatibility. Earlier versions fail on `np.float_` removal.
- **Gradio 5.x**: Chatbot message format changed to dict with `role`/`content` keys (not lists).
- **PyYAML ≥6.0.1**: Version 6.0 has build issues with Python 3.12. Use 6.0.1+ for pre-built wheels.
- **usage.py**: Deferred to Phase 3. OpenAI import path needs updating for current library version.

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

CREATE TABLE flashcards (
    id INTEGER PRIMARY KEY,
    class_id INTEGER NOT NULL REFERENCES classes(id),
    input_id INTEGER REFERENCES inputs(id),
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

### ChromaDB Structure

- One collection per class: `class_{class_id}`
- Chunk metadata includes: `source` (input name), `chunk_idx`
- Delete chunks by filtering: `collection.delete(where={"source": input_name})`

## Key Implementation Patterns

### 1. Deterministic Ingestion Pipeline

File routing is based on extensions — no agent reasoning:

```python
def process_upload(file_path: str, class_id: int, input_name: str, input_type: str):
    ext = Path(file_path).suffix.lower()

    # Route to appropriate extractor
    if ext in ('.txt', '.md'):
        raw_text = extract_plain_text(file_path)
    elif ext == '.pdf':
        raw_text = process_pdf(file_path, input_type)
    elif ext in ('.mp3', '.mp4', '.wav'):
        raw_text = whisper_transcribe(file_path)
    # ... etc

    # Chunk and embed (deterministic)
    chunks = chunk_text(raw_text, chunk_size=800, overlap=150)
    embed_chunks(class_id, input_name, chunks)

    # Store metadata
    save_input(class_id, input_name, input_type, file_path, raw_text)
```

### 2. RAG Chat Agent with Tool Use

The chat agent has access to two tools:
- `search_class_materials`: Semantic search via ChromaDB
- `execute_python`: Code-as-tool for calculations/visualizations

```python
CHAT_TOOLS = [
    {
        "name": "search_class_materials",
        "description": "Search uploaded class materials using semantic similarity",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "n_results": {"type": "integer", "default": 5}
            }
        }
    },
    {
        "name": "execute_python",
        "description": "Execute Python code for calculations or data analysis",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string"}
            }
        }
    }
]
```

Agent loop continues until the model produces a final text response (no more tool calls).

### 3. Flashcard Generation with Structured Outputs

Uses OpenAI Structured Outputs to guarantee valid JSON:

```python
response = client.responses.create(
    model="gpt-5-mini",
    input=[
        {"role": "system", "content": "Generate flashcards from art history content..."},
        {"role": "user", "content": f"Generate flashcards from:\n\n{context}"}
    ],
    text={
        "format": {
            "type": "json_schema",
            "name": "flashcard_set",
            "schema": {
                "type": "object",
                "properties": {
                    "flashcards": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "term": {"type": "string"},
                                "definition": {"type": "string"}
                            }
                        }
                    }
                }
            }
        }
    }
)
```

### 4. Code Execution Sandbox

For the `execute_python` tool, use restricted `exec()` with whitelisted modules:

```python
ALLOWED_MODULES = {"math", "numpy", "sympy", "statistics", "pandas", "matplotlib"}

def execute_python_sandboxed(code: str) -> str:
    stdout = io.StringIO()
    local_vars = {}

    for mod in ALLOWED_MODULES:
        try:
            local_vars[mod] = __import__(mod)
        except ImportError:
            pass

    try:
        with contextlib.redirect_stdout(stdout):
            exec(code, {"__builtins__": {}}, local_vars)
        return stdout.getvalue() or "Code executed successfully"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"
```

For production, consider Docker isolation or Pyodide (Python in WebAssembly).

### 5. Artwork Image Fetching (Deterministic)

**Problem**: Art history flashcards without images are significantly less effective, since visual recognition is the core skill being tested.

**Solution**: Most artworks in art history surveys are public domain. Free museum APIs provide high-quality images:

```python
import requests

def fetch_artwork_image(title: str, artist: str) -> str | None:
    """Deterministically fetch a public domain artwork image URL."""
    # Try Wikimedia Commons first
    search_url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": f"{title}",
        "prop": "pageimages",
        "pithumbsize": 400,
        "format": "json"
    }
    response = requests.get(search_url, params=params).json()
    pages = response["query"]["pages"]
    for page in pages.values():
        if "thumbnail" in page:
            return page["thumbnail"]["source"]

    # Fallback: Met Museum API
    search = requests.get(
        f"https://collectionapi.metmuseum.org/public/collection/v1/search",
        params={"q": f"{title} {artist}", "hasImages": True}
    ).json()
    if search["total"] > 0:
        obj_id = search["objectIDs"][0]
        obj = requests.get(
            f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{obj_id}"
        ).json()
        return obj.get("primaryImageSmall")

    return None  # Graceful fallback to text-only card
```

**Pipeline integration**:
```
LLM generates flashcard → fetch_artwork_image(term, artist) → store image_url in SQLite
```

Available APIs (all free, no authentication required for basic use):
- **Wikimedia Commons**: Virtually all artworks
- **Met Museum Open Access**: 400K+ works
- **Art Institute of Chicago**: 100K+ works
- **Europeana**: European cultural heritage (free API key)

This is a **post-MVP feature** — simple deterministic function, no agent logic needed. Adds significant value for art history use case.

## Domain Specialization: Art History

The system is tuned for art history content:

**Flashcard format**: Artist, Title, Date, Period, Medium, Style, Significance

**System prompt for chat agent**:
```
You are an expert art history study assistant with deep knowledge of
Western art from antiquity through contemporary periods.

When answering questions:
1. Always ground your response in the student's uploaded course materials.
2. Use proper art historical terminology (chiaroscuro, sfumato, tenebrism, etc.)
3. When discussing an artwork, reference: artist, title, date, medium if known.
4. Cite which lecture/reading the information comes from.
```

**Chunking strategy**: 800-token chunks with 150-token overlap, preserving artwork contexts.

## MVP Build Order

1. **Foundation**: SQLite schema, ChromaDB initialization, file uploads
2. **Ingestion Pipeline**: PDF extraction, chunking, embedding
3. **RAG Chat Agent**: Tool definitions, agent loop, Gradio `ChatInterface`
4. **Flashcard Generation**: Structured outputs, CSV export
5. **UI Assembly**: Gradio `Blocks` layout with upload | chat | flashcards panels

Post-MVP: Quiz generation, reflection loop, artwork image fetching, multi-class dashboard, Whisper transcription, Mathpix OCR, Flask migration.

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
python gradio_app.py      # Launch Gradio UI at http://127.0.0.1:7860
```

**Testing (Phase 2+):**
```bash
pytest  # Run tests
```

## API Keys Required

The following environment variables should be set in `.env`:

- `OPENAI_API_KEY`: For GPT models, Whisper, and embeddings
- `ANTHROPIC_API_KEY`: For Claude tool use agent
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
- **Gradio MVP**: Using Gradio 5.x for rapid prototyping. Flask + Tailwind CSS is the post-MVP production version.
- **Code execution is post-MVP**: The `execute_python` tool is only needed when expanding to CS content that requires computation.
- **Virtual environment required**: Always activate venv before running commands. Dependencies are isolated from system Python.
- **Usage tracking deferred**: `usage.py` from course materials needs OpenAI import path update. Will be re-enabled in Phase 3.

## Phase 2 Implementation Summary

Phase 2 is now complete. Here's what was implemented:

1. **Files created:**
   - `app/pipelines/ingestion.py` - Deterministic file routing and text extraction ✅
     - `extract_plain_text()` for .txt/.md files
     - `extract_docx()` for DOCX files
     - `extract_pdf()` for PDF files using pypdf
     - `process_upload()` main entry point

   - `app/pipelines/chunking.py` - Text splitting and embedding ✅
     - `chunk_text()` using RecursiveCharacterTextSplitter
     - `generate_embeddings()` with OpenAI text-embedding-3-small
     - `delete_embeddings()` for cleanup when files are removed

2. **Integration completed:**
   - Updated `gradio_app.py` `upload_file()` to call full pipeline
   - Populates `Input.raw_text` field with extracted text
   - Uses `get_or_create_collection()` from extensions.py
   - Stores chunks with metadata: `{"source": input_name, "chunk_idx": i}`
   - Upload status now shows detailed ingestion progress

3. **Configuration used:**
   - `CHUNK_SIZE = 800` tokens
   - `CHUNK_OVERLAP = 150` tokens
   - `EMBEDDING_MODEL = 'text-embedding-3-small'`
   - Semantic separators: `["\n\n", "\n", ". ", " "]`

## Additional Features Implemented

### File Management (2026-03-20)
**Purpose**: Enable users to manage uploaded files and clean up test data

**Features:**
- **Individual File Deletion**: Select and delete specific uploaded files with confirmation
- **Bulk Delete**: "Clear All Files" button to remove all files from a class (for testing)
- **Data Cleanup**: Automatically removes files from disk, SQLite, and ChromaDB
- **Cascade Deletion**: Related flashcards are removed when source file is deleted
- **Chat History Preserved**: Chat messages remain intact as conversational context

**Implementation** (gradio_app.py):
- `delete_file(input_id, class_name)` - Transaction-safe deletion across 3 storage layers
- `clear_all_files(class_name)` - Bulk deletion with best-effort cleanup
- `get_current_file_list(class_name)` - Helper for UI file list display
- `get_file_choices(class_name)` - Helper for file selector dropdown

**UI Components**:
- File selector dropdown in upload panel
- "Delete Selected" button with confirmation modal
- "Clear All Files" button with warning confirmation
- Real-time file list updates after operations

### Class Switching (2026-03-20)
**Purpose**: Improve UX by allowing users to easily switch between existing classes

**Features:**
- **Dropdown Selector**: Replace text input with dropdown showing all existing classes
- **Create New Class**: "➕ Create New Class..." option with conditional text input
- **Auto-Update**: Dropdown refreshes when new classes are created
- **Context Switching**: File list and all UI elements update when switching classes
- **No Default Selection**: Forces explicit class selection for clarity

**Implementation** (gradio_app.py):
- `get_all_classes()` - Query all classes, return sorted list + create option
- `handle_class_selection(selected_value)` - Show/hide new class input, load files
- `create_new_class(new_class_name)` - Create class, update dropdown, switch context
- Hidden `current_class_name` state component for tracking actual class name

**UI Pattern**:
- `class_selector` dropdown (populated on load, updated after uploads)
- `new_class_input` textbox (hidden until "Create New Class" selected)
- `create_class_btn` button (hidden until needed)
- All event handlers updated to use hidden state instead of direct textbox

**Benefits:**
- **Discoverability**: Users see all available classes without guessing names
- **Error Prevention**: Dropdown prevents typos that would create duplicate classes
- **Faster Workflow**: One-click switching vs. typing entire class name
- **Auto-Sync**: New classes appear immediately without page refresh

## Starting Phase 3

When ready to implement the RAG chat agent:

1. **Files to create:**
   - `app/agents/chat_agent.py` - RAG agent with tool use
   - Update `gradio_app.py` - Wire chat interface to agent

2. **Tools to implement:**
   - `search_class_materials` - ChromaDB semantic search
   - `execute_python` - Code execution sandbox (post-MVP for CS domain)

3. **Key components:**
   - Agent loop using course `run_agent.py` pattern
   - Chat history persistence to `chat_messages` table
   - Integration with Gradio `gr.Chatbot` component

4. **Prerequisites (already complete):**
   - ✅ ChromaDB collections populated with embeddings
   - ✅ Metadata includes source attribution
   - ✅ Agent execution framework from course materials
