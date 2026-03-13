# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

StudyAgent is an AI-powered study tool for art history courses that demonstrates a **hybrid architecture** — deterministic code paths where tasks are predictable, agentic LLM-driven processes where reasoning adds value. The core principle: "if the logic fits in a flowchart, it doesn't need an agent."

Key features:
- Upload lecture PDFs/documents → deterministic OCR and text extraction
- Chat with class materials → RAG agent with tool use (search + code execution)
- Generate flashcards → structured LLM outputs with term/definition pairs
- Export to Quizlet/Anki → deterministic CSV formatting

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
│   ├── __init__.py              # Flask factory
│   ├── config.py                # API keys, DB paths
│   ├── extensions.py            # SQLAlchemy + ChromaDB init
│   ├── models/                  # SQLAlchemy ORM models
│   ├── pipelines/               # ⚡ DETERMINISTIC LAYER
│   │   ├── ingestion.py         # File routing + extraction
│   │   ├── ocr.py               # Mathpix + Claude vision
│   │   ├── transcription.py     # Whisper wrapper
│   │   ├── chunking.py          # Text splitting + embedding
│   │   └── exporters.py         # CSV/TSV for Quizlet/Anki
│   ├── agents/                  # 🤖 AGENTIC LAYER
│   │   ├── chat_agent.py        # RAG + code-as-tool
│   │   ├── study_agent.py       # Flashcard/quiz generation
│   │   ├── tools.py             # Tool definitions + dispatch
│   │   └── sandbox.py           # Code execution sandbox
│   ├── routes/                  # Flask Blueprints
│   ├── templates/
│   └── static/
├── data/
│   ├── uploads/                 # User-uploaded files
│   ├── app.db                   # SQLite metadata
│   └── chroma/                  # ChromaDB vector store
```

## Tech Stack

| Component | Technology | Notes |
|-----------|-----------|-------|
| Backend | Flask + SQLAlchemy | Python-native, modular with Blueprints |
| Metadata DB | SQLite | Zero-config, single-file database |
| Vector DB | ChromaDB | One collection per class, in-process |
| LLM APIs | OpenAI + Anthropic | OpenAI for structured outputs, Claude for tool use |
| Math OCR | Mathpix API | Handles equations, diagrams, STEM content |
| Audio | OpenAI Whisper | Lecture recording transcription |
| Embeddings | text-embedding-3-small | Fast, cost-effective |
| Frontend (MVP) | Gradio Blocks | Rapid prototyping |
| Frontend (v2) | Tailwind CSS + HTMX | Production UI, no React needed |

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

Post-MVP: Quiz generation, reflection loop, multi-class dashboard, Whisper transcription, Mathpix OCR, Flask migration.

## Development Commands

Since the codebase is still in design phase, these will be added as development progresses:

- TBD: Run development server
- TBD: Initialize database
- TBD: Run tests
- TBD: Generate embeddings for uploaded files

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
- **Start with Gradio, migrate to Flask**: The MVP uses Gradio for rapid prototyping. Flask + Tailwind CSS is the post-MVP production version.
- **Code execution is post-MVP**: The `execute_python` tool is only needed when expanding to CS content that requires computation.
- **Cost tracking**: Use `usage.py` pattern from course materials to track LLM API costs during development.
