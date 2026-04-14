# Agentic Study Tool — Design Architecture for CS 301R

## Whiteboard Summary

The whiteboard brainstorm outlines an **agentic-powered study website** where students create classes, upload academic materials (slides, recordings, textbooks, lecture notes, homework, quizzes), and leverage AI to generate flashcards, sample quizzes, and a class-centered chat interface. Materials flow through OCR/transcription pipelines, get embedded into a RAG knowledge store, and produce study aids.

---

## Design Philosophy: Hybrid Architecture

Anthropic's own guidance on building effective agents is clear: **"find the simplest solution possible, and only increase complexity when needed"**. Agentic systems trade latency and cost for flexibility — a single user request can expand into N planner calls, M executor calls, and reflection passes, with costs scaling at O(N × M) rather than O(1). For a class project, this means being intentional about where agents add value versus where deterministic code is faster, cheaper, and more reliable.[^1][^2][^3]

The rule of thumb: **if the logic fits in a flowchart, it doesn't need an agent**. Use deterministic workflows for tasks that are predictable, repeatable, and follow a single clear path. Use agents where the path cannot be anticipated, inputs vary significantly, or the task requires reasoning and adaptation.[^4][^5][^6]

This hybrid approach is actually the most impressive thing you can demonstrate in CS 301R — it shows you understand _when_ to use agents, not just _how_.

### The Decision Matrix

| Task                                         | Approach                      | Why                                                                                                                |
| -------------------------------------------- | ----------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| File type routing (PDF → OCR, MP3 → Whisper) | **Deterministic**             | File extensions are known; a simple `if/else` is faster, cheaper, and 100% reliable[^5][^7]                        |
| Mathpix OCR API call                         | **Deterministic**             | Fixed API call with fixed parameters — no reasoning needed[^8]                                                     |
| Whisper transcription                        | **Deterministic**             | Same — send audio, get text back[^9]                                                                               |
| Chunking & embedding text                    | **Deterministic**             | Same chunking strategy every time; no judgment required[^10]                                                       |
| CSV export for Quizlet/Anki                  | **Deterministic**             | Fixed format transformation, zero ambiguity[^11]                                                                   |
| Chat with class materials (RAG)              | **Agent (RAG + Tool Use)**    | Requires retrieval reasoning, context selection, and synthesis[^2]                                                 |
| Generating flashcards from varied content    | **Agent (with code-as-tool)** | Content varies wildly; agent must judge what's "card-worthy," handle different formats, and produce quality output |
| Generating quizzes from content              | **Agent (with reflection)**   | Requires judgment about difficulty, question types, and answer quality                                             |
| "Help me study Chapter 5" (multi-step)       | **Agent (Planning)**          | Open-ended request requiring decomposition into subtasks[^6]                                                       |

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                    FRONTEND (Website UI)                       │
│   Class Dashboard  •  Upload UI  •  Chat  •  Flashcards      │
│   Quiz Interface   •  Study Guide Viewer                      │
└──────────────┬────────────────────────────────────────────────┘
               │  REST API
┌──────────────▼────────────────────────────────────────────────┐
│                      FLASK BACKEND                            │
│                                                               │
│  ┌─────────────────────────┐  ┌────────────────────────────┐  │
│  │  DETERMINISTIC LAYER    │  │  AGENTIC LAYER             │  │
│  │  (Predefined code paths)│  │  (LLM-directed processes)  │  │
│  │                         │  │                            │  │
│  │  • File type routing    │  │  • RAG Chat Agent          │  │
│  │  • Mathpix OCR calls    │  │  • Study Generation Agent  │  │
│  │  • Whisper transcription│  │    (flashcards, quizzes)   │  │
│  │  • Text extraction      │  │  • Code-as-Tool execution  │  │
│  │  • Chunking & embedding │  │  • Reflection/self-critique│  │
│  │  • CSV/TSV export       │  │                            │  │
│  │  • DB CRUD operations   │  │                            │  │
│  └─────────────────────────┘  └────────────────────────────┘  │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │               DATA LAYER                                │  │
│  │  SQLite (metadata, classes, inputs, flashcards, quizzes)│  │
│  │  ChromaDB (vector embeddings per class)                 │  │
│  └─────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

The split is explicit: the **left side** is hardcoded Python functions that always run the same way. The **right side** is where Claude makes decisions at runtime. This mirrors Anthropic's own distinction — workflows use "predefined code paths," agents "dynamically direct their own processes".[^12][^2]

---

## Deterministic Layer: Ingestion Pipeline

These are pure Python functions with no LLM calls. They're fast, free, and deterministic.

### File Routing (No Agent Needed)

```python
def process_upload(file_path: str, class_id: int, input_name: str, input_type: str):
    """Deterministic routing based on file extension + user-provided type."""
    ext = Path(file_path).suffix.lower()

    # Step 1: Extract text (deterministic routing)
    if ext in ('.txt', '.md'):
        raw_text = extract_plain_text(file_path)
    elif ext == '.docx':
        raw_text = extract_docx(file_path)
    elif ext in ('.jpg', '.jpeg', '.png'):
        raw_text = mathpix_ocr(file_path)  # Always use Mathpix for images
    elif ext == '.pdf':
        raw_text = process_pdf(file_path, input_type)
    elif ext in ('.mp3', '.mp4', '.wav', '.m4a'):
        raw_text = whisper_transcribe(file_path)
    else:
        raise UnsupportedFileType(ext)

    # Step 2: Chunk and embed (deterministic)
    chunks = chunk_text(raw_text, chunk_size=800, overlap=150)
    embed_chunks(class_id, input_name, chunks)

    # Step 3: Store metadata (deterministic)
    save_input(class_id, input_name, input_type, file_path, raw_text)

    # Step 4: Generate summary (single LLM call — a workflow, not an agent)
    summary = generate_summary(raw_text)
    update_input_summary(input_name, summary)

    return raw_text
```

Why no agent here? The whiteboard notes "if math/STEM → use Mathpix, otherwise → use Claude 3.5 Sonnet for OCR." That's a flowchart — it has exactly two branches with a clear condition. An agent would burn 3–6 extra LLM calls reasoning about something you can determine from the file extension and a checkbox. A single Mathpix API call costs fractions of a cent; wrapping it in an agent reasoning loop could cost 10–100x more.[^1][^13][^14][^15]

### Mathpix OCR (Deterministic Tool)

Mathpix reads handwritten and printed STEM content — math equations, tables, chemistry diagrams — outputting clean LaTeX and Mathpix Markdown. It's called as a direct API wrapper:[^16][^8]

```python
import requests

def mathpix_ocr(file_path: str) -> str:
    """Direct Mathpix API call — no agent reasoning needed."""
    with open(file_path, "rb") as f:
        response = requests.post(
            "https://api.mathpix.com/v3/text",
            files={"file": f},
            headers={"app_id": MATHPIX_APP_ID, "app_key": MATHPIX_APP_KEY},
            data={"options_json": json.dumps({
                "formats": ["text", "latex_styled"],
                "math_inline_delimiters": ["\\(", "\\)"],
            })}
        )
    return response.json().get("text", "")
```

### Whisper Transcription (Deterministic Tool)

OpenAI Whisper converts lecture recordings to text with high accuracy. For files over 25MB, split with `pydub` first:[^9][^17]

```python
from openai import OpenAI

def whisper_transcribe(file_path: str) -> str:
    """Direct Whisper API call — deterministic."""
    client = OpenAI()
    with open(file_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="whisper-1", file=f
        )
    return transcript.text
```

### Chunking & Embedding (Deterministic)

Semantic-aware chunking with 800-token chunks and 150-token overlap works well for educational content. ChromaDB is the vector store — open-source, in-process, persists to disk:[^18][^19][^10][^20]

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800, chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " "]
)
chroma_client = chromadb.PersistentClient(path="./data/chroma")

def chunk_text(text: str, chunk_size=800, overlap=150) -> list[str]:
    return text_splitter.split_text(text)

def embed_chunks(class_id: int, source_name: str, chunks: list[str]):
    collection = chroma_client.get_or_create_collection(f"class_{class_id}")
    collection.add(
        documents=chunks,
        ids=[f"{source_name}_{i}" for i in range(len(chunks))],
        metadatas=[{"source": source_name, "chunk_idx": i} for i in range(len(chunks))]
    )
```

### CSV Export (Deterministic)

Quizlet accepts tab-separated imports; Anki uses CSV with pipe delimiters. This is a pure formatting function:[^11][^21][^22]

```python
def export_flashcards(flashcards: list[dict], format: str = "quizlet") -> str:
    if format == "quizlet":
        return "\n".join(f"{fc['term']}\t{fc['definition']}" for fc in flashcards)
    elif format == "anki":
        return "\n".join(f"{fc['term']}|{fc['definition']}" for fc in flashcards)
```

---

## Agentic Layer: Where Agents Earn Their Keep

These are the tasks where inputs vary, the path can't be predicted, and reasoning adds genuine value.[^4][^2]

### Agentic Pattern 1: RAG Chat Agent (Tool Use)

The chat interface is where students ask free-form questions about their class materials. This requires an agent because:

- The query could be anything — factual recall, conceptual explanation, problem-solving
- The agent must decide _how many_ chunks to retrieve, _which_ to prioritize, and _how_ to synthesize them
- Follow-up questions require maintaining conversational context

The agent has access to tools:

```python
CHAT_TOOLS = [
    {
        "name": "search_class_materials",
        "description": "Search the student's uploaded class materials using semantic "
                       "similarity. Returns the most relevant text chunks.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "n_results": {"type": "integer", "description": "Number of chunks to retrieve", "default": 5}
            },
            "required": ["query"]
        }
    },
    {
        "name": "execute_python",
        "description": "Execute Python code to perform calculations, solve math problems, "
                       "create visualizations, or process data. Use for any task requiring "
                       "computation rather than text generation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"}
            },
            "required": ["code"]
        }
    }
]
```

The **code-as-tool** (called `execute_python` above) is key. When a student asks "solve this integral" or "plot the frequency distribution from Lecture 3's dataset," the agent can write and execute Python code rather than trying to compute in its head. This is the **CodeAct pattern** — the agent uses code execution as its primary action mechanism, which is more reliable than pure text reasoning for computational tasks.[^23][^24][^25][^26]

#### Code-as-Tool Implementation

The sandbox doesn't need to be complex for a class project. A restricted `exec()` with whitelisted libraries works:

```python
import io, contextlib

ALLOWED_MODULES = {"math", "numpy", "sympy", "statistics", "pandas", "matplotlib"}

def execute_python_sandboxed(code: str) -> str:
    """Execute Python code in a restricted environment."""
    stdout = io.StringIO()
    local_vars = {}

    # Pre-import allowed modules
    for mod in ALLOWED_MODULES:
        try:
            local_vars[mod] = __import__(mod)
        except ImportError:
            pass

    try:
        with contextlib.redirect_stdout(stdout):
            exec(code, {"__builtins__": {}}, local_vars)
        output = stdout.getvalue()
        return output if output else "Code executed successfully (no output)."
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"
```

For a more robust approach, Docker container isolation or Pyodide (Python in WebAssembly) provide stronger sandboxing.[^25]

#### Agent Loop

```python
import anthropic

client = anthropic.Anthropic()

def chat_agent(class_id: int, question: str, history: list[dict]) -> str:
    """RAG Chat Agent with tool use — the LLM decides what to do."""
    system = (
        "You are a study assistant with access to the student's class materials. "
        "Use search_class_materials to find relevant content before answering. "
        "Use execute_python for any calculations, math solving, or data analysis. "
        "Always ground your answers in the retrieved materials."
    )

    messages = history + [{"role": "user", "content": question}]

    # Agent loop — LLM decides when to call tools and when to stop
    while True:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            system=system,
            tools=CHAT_TOOLS,
            messages=messages
        )

        if response.stop_reason == "tool_use":
            # Agent chose to use a tool
            tool_block = next(b for b in response.content if b.type == "tool_use")
            result = dispatch_tool(tool_block.name, tool_block.input, class_id)

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": tool_block.id, "content": result}
            ]})
        else:
            # Agent decided it has enough info to answer
            return next(b.text for b in response.content if hasattr(b, "text"))


def dispatch_tool(name: str, inputs: dict, class_id: int) -> str:
    if name == "search_class_materials":
        collection = chroma_client.get_collection(f"class_{class_id}")
        results = collection.query(query_texts=[inputs["query"]], n_results=inputs.get("n_results", 5))
        return "\n\n---\n\n".join(results["documents"])
    elif name == "execute_python":
        return execute_python_sandboxed(inputs["code"])
```

This is a true agent — the LLM decides at runtime whether to search materials, run code, or answer directly. It can chain multiple tool calls (search → compute → respond). The loop continues until the model produces a final text response.[^27][^2]

### Agentic Pattern 2: Flashcard Generation (Post-MVP: Planning + Reflection)

> **Implementation Status (Phase 4+):** Flashcard generation evolved beyond the MVP design below. The current implementation uses a full RAG agent with tool use (`list_sections` + `search_class_materials` + `generate_flashcards_structured`) rather than a single structured LLM call. Key additions: content-driven count (generates one flashcard per matching item, no user-specified count), post-processing category filter (LLM classifies and removes off-category items with fail-open logic), and cancellable generation via `threading.Event`. See `app/agents/study_agent.py`.

For the **MVP**, flashcard generation uses a simpler approach — retrieve relevant chunks via ChromaDB, then make a **single structured LLM call** to produce term/definition pairs. This is technically a workflow (predefined code path), not a full agent, and that's the right call for an MVP.[^2]

This uses **OpenAI Structured Outputs** to guarantee valid JSON matching the flashcard schema — no parsing or retry logic needed:[^28][^29]

```python
def generate_flashcards(class_id: int, topic: str = None) -> list[dict]:
    """MVP: retrieve + single structured LLM call. Not a full agent yet."""
    collection = chroma_client.get_collection(f"class_{class_id}")
    query = topic or "key concepts and terms"
    results = collection.query(query_texts=[query], n_results=10)
    context = "\n\n".join(results["documents"])

    response = client.responses.create(
        model="gpt-5-mini",
        input=[
            {"role": "system", "content":
                "Generate flashcards from art history content. "
                "Focus on: artist names, artwork titles, periods, movements, "
                "techniques, and key concepts."},
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
                                },
                                "required": ["term", "definition"]
                            }
                        }
                    },
                    "required": ["flashcards"]
                }
            }
        }
    )
    return json.loads(response.output_text)["flashcards"]
```

This pattern matches the YAML-based structured output configs from Lecture 3a. The `json_schema` format guarantees every response is a valid array of `{term, definition}` objects — no parsing errors, no retries.[^30]

**Post-MVP upgrades** to this component:

- **Planning**: Agent decides _which_ chunks to retrieve for a given study request (uses tool calling)
- **Reflection**: Evaluator-optimizer loop where a second LLM call critiques the flashcards for duplicates, missing concepts, and accuracy. Cap at 2 iterations to control cost.[^31][^2]
- **Quiz generation**: Same pattern but outputting multiple-choice questions instead of term/definition pairs

---

## Artwork Image Fetching (Deterministic, Art History Enhancement)

**The Problem**: Art history flashcards without images are significantly less effective, since the whole skill being tested is visual recognition. Text-only flashcards miss the core learning objective.

**The Solution**: Most artworks taught in undergrad art history surveys (Renaissance, Baroque, Gothic, etc.) are centuries old and their images are fully in the public domain. Several museums publish free APIs specifically for this.

### Public Domain Art APIs

| API                          | Collection                 | Cost | Key Required |
| ---------------------------- | -------------------------- | ---- | ------------ |
| **Wikimedia Commons**        | Virtually everything       | Free | No           |
| **Met Museum Open Access**   | 400K+ works                | Free | No           |
| **Art Institute of Chicago** | 100K+ works                | Free | No           |
| **Europeana**                | European cultural heritage | Free | Yes (free)   |

This means flashcard image fetching is a **deterministic pipeline step**, not an agentic one — just a function call with the artwork title:

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

    return None  # Fall back to text-only card
```

### How It Fits the Architecture

When the LLM generates a flashcard with a structured output like `{"term": "Portinari Altarpiece", "definition": "Hugo van der Goes, c.1475..."}`, the ingestion pipeline immediately tries to attach an image:

```
LLM generates flashcard → deterministic image fetch → store URL in SQLite
```

The flashcard then renders with the image on one side and the term/definition on the other. If no image is found, it gracefully falls back to text-only — so the feature never breaks the app.

### Why This Is Better Than Your Professor's Slides

Your professor's slides are low-res JPEGs in a PDF. Wikimedia Commons and the Met API return high-resolution, properly lit, color-accurate images — often better than what you'd see in class.

This is a strong **post-MVP feature** but it's simple enough (one deterministic function, no new agent logic) that it could slot into Phase 4 alongside flashcard generation with minimal extra effort.

---

## Post-MVP: Code-as-Tool (CS Domain Expansion)

When expanding to computer science, the agent gains a **code execution tool** — instead of reasoning about math/code in text, it writes and runs Python for deterministic results. This is a post-MVP feature since art history doesn't need computation. It's documented in the Agentic Layer section above for when you're ready to add it.[^23][^25]

---

## Data Layer

### SQLite Schema

SQLite is the right metadata store — zero-config, portable, single-file:[^32]

```sql
CREATE TABLE classes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE inputs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    class_id    INTEGER NOT NULL REFERENCES classes(id) ON DELETE CASCADE,
    name        TEXT NOT NULL,        -- user-given name (deletable per whiteboard)
    input_type  TEXT NOT NULL,        -- 'slides','recording','notes','textbook','quiz'
    file_path   TEXT,
    raw_text    TEXT,
    summary     TEXT,                 -- auto-generated summary
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE flashcards (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    class_id    INTEGER NOT NULL REFERENCES classes(id) ON DELETE CASCADE,
    input_id    INTEGER REFERENCES inputs(id) ON DELETE SET NULL,
    term        TEXT NOT NULL,
    definition  TEXT NOT NULL,
    image_url   TEXT,                 -- public domain artwork image (optional)
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE quizzes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    class_id    INTEGER NOT NULL REFERENCES classes(id) ON DELETE CASCADE,
    question    TEXT NOT NULL,
    options     TEXT,                 -- JSON array for MC options
    answer      TEXT NOT NULL,
    explanation TEXT,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE chat_messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    class_id    INTEGER NOT NULL REFERENCES classes(id) ON DELETE CASCADE,
    role        TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
    content     TEXT NOT NULL,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### ChromaDB (Vector Store)

One ChromaDB collection per class. When an input is deleted, its chunks are removed via metadata filtering:

```python
collection.delete(where={"source": input_name})
```

---

## Tech Stack

| Layer            | Technology                      | Rationale                                |
| ---------------- | ------------------------------- | ---------------------------------------- |
| **Frontend**     | HTML/CSS/JS + Jinja2            | Simplest path for a class project[^33]   |
| **Backend**      | Flask + SQLAlchemy + Blueprints | Python-native, modular[^34]              |
| **Metadata DB**  | SQLite                          | Zero-config, single-file[^32]            |
| **Vector DB**    | ChromaDB                        | Open-source, in-process, persistent[^19] |
| **LLM**          | Claude API (tool use)           | Native tool calling for agent loops[^2]  |
| **Math OCR**     | Mathpix API                     | Best for equations and STEM[^8]          |
| **Audio**        | OpenAI Whisper API              | Accurate transcription[^9]               |
| **Embeddings**   | `all-MiniLM-L6-v2`              | Free, fast, local                        |
| **Code sandbox** | Restricted `exec()` or Docker   | For code-as-tool execution[^25]          |

---

## Project Structure

```
study_agent/
├── app/
│   ├── __init__.py              # Flask factory
│   ├── config.py                # API keys, DB paths
│   ├── extensions.py            # SQLAlchemy + ChromaDB init
│   │
│   ├── models/                  # SQLAlchemy ORM models
│   │   ├── class_model.py
│   │   ├── input_model.py
│   │   ├── flashcard_model.py
│   │   ├── quiz_model.py
│   │   └── chat_model.py
│   │
│   ├── pipelines/               # ⚡ DETERMINISTIC LAYER
│   │   ├── ingestion.py         # File routing + extraction
│   │   ├── ocr.py               # Mathpix + Claude vision wrappers
│   │   ├── transcription.py     # Whisper wrapper
│   │   ├── chunking.py          # Text splitting + embedding
│   │   ├── artwork_images.py    # Public domain image fetching
│   │   └── exporters.py         # CSV/TSV for Quizlet/Anki
│   │
│   ├── agents/                  # 🤖 AGENTIC LAYER
│   │   ├── chat_agent.py        # RAG + code-as-tool
│   │   ├── study_agent.py       # Planning + reflection
│   │   ├── tools.py             # Tool definitions + dispatch
│   │   └── sandbox.py           # Code execution sandbox
│   │
│   ├── routes/                  # Flask Blueprints (thin REST)
│   │   ├── classes.py
│   │   ├── inputs.py
│   │   ├── chat.py
│   │   ├── flashcards.py
│   │   └── quizzes.py
│   │
│   ├── templates/
│   └── static/
│
├── data/
│   ├── uploads/
│   ├── app.db
│   └── chroma/
│
├── requirements.txt
└── run.py
```

The key structural choice: `pipelines/` (deterministic) and `agents/` (agentic) are **separate directories**. This makes the architectural distinction visible in the codebase, which is excellent for a class presentation.

---

## Agentic Patterns Summary

For your CS 301R writeup/presentation, here are the four agentic design patterns demonstrated and _why_ each is used only where it belongs:[^35][^36]

| Pattern                              | Component                | Why Agent (Not Deterministic)                                                                              |
| ------------------------------------ | ------------------------ | ---------------------------------------------------------------------------------------------------------- |
| **Tool Use** (ReAct)                 | Chat Agent               | Student questions are open-ended; agent must decide whether to search, compute, or answer directly[^37]    |
| **Code-as-Tool**                     | Chat Agent + Study Agent | Math, data analysis, and formatting tasks are more reliable when computed than reasoned about in text[^23] |
| **Planning** (Plan-Act)              | Study Generation Agent   | "Help me study X" is an open-ended request that requires decomposition[^38]                                |
| **Reflection** (Evaluator-Optimizer) | Study Generation Agent   | Flashcard/quiz quality improves measurably with a self-critique loop[^39][^40]                             |

And critically, everything else is **deliberately not agentic** — file routing, OCR calls, transcription, chunking, embedding, CSV export, and CRUD operations are all deterministic Python. This demonstrates the Anthropic principle: "workflows offer predictability and consistency for well-defined tasks, whereas agents are the better option when flexibility and model-driven decision-making are needed".[^2]

---

## Cost Comparison

To quantify the hybrid approach's value:

| Operation                            | Agentic Cost                          | Deterministic Cost              | Savings                         |
| ------------------------------------ | ------------------------------------- | ------------------------------- | ------------------------------- |
| Route a PDF to Mathpix               | ~$0.01–0.05 (3–6 LLM calls to reason) | ~$0.00 (if/else)                | 100%                            |
| Transcribe 10-min lecture            | ~$0.10+ (agent reasoning + Whisper)   | ~$0.06 (Whisper only)           | ~40%                            |
| Chunk + embed 5 pages                | ~$0.05+ (agent overhead)              | ~$0.001 (code only)             | ~98%                            |
| Answer a chat question (RAG)         | ~$0.02–0.05 (agent + retrieval)       | N/A — requires reasoning        | Agent justified                 |
| Generate 20 flashcards w/ reflection | ~$0.08–0.15 (generate + reflect)      | ~$0.03 (single LLM call, no QC) | Agent earns its cost in quality |

Per Anthropic, the cost tradeoff is justified when "flexibility and model-driven decision-making are needed" — which is exactly the chat and study generation features, but _not_ the ingestion pipeline.[^3]

---

## Making It Demo-Ready and Resume-Worthy

The difference between a class project and a portfolio piece is **polish, presentation, and proof it works**. Hiring managers reviewing AI portfolios want to see end-to-end implementation — data preprocessing, pipeline orchestration, model logic, API integrations, and cloud deployment — not just a Jupyter notebook. Projects with comprehensive READMEs receive 3x more GitHub stars and 5x more contributor engagement. Here's how to get there.[^41][^42][^43]

### Frontend: Tailwind CSS + HTMX (No React Needed)

For a Flask project where the backend is the star, avoid heavy frontend frameworks. **Tailwind CSS via CDN + HTMX** gives a polished, modern UI with zero build step.[^44][^45]

**Tailwind CSS** provides utility-first styling — dark mode support, responsive design, and professional aesthetics out of the box. Load it from CDN in your base template:[^46]

```html
<!-- base.html -->
<!DOCTYPE html>
<html lang="en" class="dark">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>StudyAgent - {% block title %}{% endblock %}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/htmx.org@1.9.10"></script>
    <script>
      tailwind.config = {
        darkMode: "class",
        theme: {
          extend: {
            colors: {
              primary: { 500: "#6366f1", 600: "#4f46e5", 700: "#4338ca" },
              surface: { 50: "#f8fafc", 800: "#1e293b", 900: "#0f172a" },
            },
          },
        },
      };
    </script>
  </head>
  <body class="bg-surface-900 text-gray-100 min-h-screen">
    {% include 'nav.html' %}
    <main class="max-w-6xl mx-auto px-4 py-8">
      {% block content %}{% endblock %}
    </main>
  </body>
</html>
```

**HTMX** enables dynamic partial-page updates without writing JavaScript — forms submit, content swaps, and loading states are all handled with HTML attributes. This is perfect for the chat interface:[^47][^48][^49]

```html
<!-- Chat input that streams responses -->
<form
  hx-post="/api/chat/{{ class_id }}"
  hx-target="#chat-messages"
  hx-swap="beforeend"
  hx-indicator="#typing-indicator"
>
  <div class="flex gap-2">
    <input
      type="text"
      name="message"
      class="flex-1 bg-surface-800 border border-gray-700 rounded-lg px-4 py-2
                      focus:ring-2 focus:ring-primary-500 focus:outline-none"
      placeholder="Ask about your class materials..."
      autocomplete="off"
    />
    <button
      class="bg-primary-600 hover:bg-primary-700 px-6 py-2 rounded-lg 
                       font-medium transition-colors"
    >
      Send
    </button>
  </div>
</form>
<div id="typing-indicator" class="htmx-indicator text-gray-400 text-sm py-2">
  <span class="animate-pulse">● Thinking...</span>
</div>
```

### MVP UI: Gradio Blocks Layout

Gradio `gr.Blocks()` gives you a working three-panel layout with minimal code. You already know this from `chatbot.py --web` in Lecture 1c:[^30]

```python
import gradio as gr

with gr.Blocks(title="StudyAgent", theme=gr.themes.Soft(primary_hue="indigo")) as app:
    gr.Markdown("# 🎨 StudyAgent — Art History")

    with gr.Row():
        # Left: Upload
        with gr.Column(scale=1):
            upload = gr.File(label="Upload PDF / Text", file_types=[".pdf", ".txt", ".docx"])
            upload_btn = gr.Button("Ingest")
            file_list = gr.Dataframe(headers=["File", "Status"], label="Uploaded Materials")

        # Center: Chat
        with gr.Column(scale=2):
            chatbot = gr.ChatInterface(fn=chat_agent_fn, title="Ask about your materials")

        # Right: Flashcards
        with gr.Column(scale=1):
            gen_btn = gr.Button("✨ Generate Flashcards")
            flashcard_table = gr.Dataframe(headers=["Term", "Definition"], label="Flashcards")
            export_btn = gr.Button("💾 Export CSV")
            csv_download = gr.File(label="Download")

app.launch()
```

This gives you a polished, functional UI in ~30 lines. Gradio handles the loading spinners, responsive layout, and state management automatically.[^50][^51]

### Post-MVP UI Upgrades

- Migrate to Flask + Tailwind CSS for full custom design (the resume-worthy version)
- Streaming token-by-token responses via Server-Sent Events
- Card flip animation for flashcards
- Multi-class dashboard
- Interactive quiz mode

---

## Post-MVP: Deployment & GitHub Presentation

These are important for the resume-quality goal but are **not part of the MVP**. Do them after the core features work.

- **Deployment**: Gradio apps can be deployed to **Hugging Face Spaces** for free (one-click from GitHub), or to Railway ($5 free tier) if you migrate to Flask later.[^52][^53]
- **GitHub README**: Hero screenshot/GIF, architecture diagram (Mermaid), "Why Hybrid?" section explaining agentic vs. deterministic split, tech stack badges, setup instructions, cost analysis using your `usage.py` data.[^30]
- **Demo video (2 min)**: Upload PDF → chat a question → generate flashcards → export CSV. Close with: "Agents only where reasoning adds value; deterministic pipelines everywhere else."

---

## MVP Build Order

## Choosing an MVP Domain

Scoping to one class type is smart — it lets you hardcode domain-specific prompt tuning, choose the right OCR pipeline, and deliver a polished demo for one use case rather than a mediocre one for three. Here's how the three candidates compare:

| Dimension               | Religion (BYU)                                                                                    | Art History                                                                                    | Computer Science                                                                           |
| ----------------------- | ------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| **Input types**         | Scripture PDFs, lecture recordings, study guides — mostly text                                    | Slides with artwork images, textbook PDFs, image-heavy handouts                                | Slides, code files, textbooks with math/diagrams                                           |
| **OCR complexity**      | Low — mostly typed text, minimal special formatting                                               | Medium — image captions and artwork labels, but text is standard                               | High — math equations, code blocks, diagrams need Mathpix[^8]                              |
| **Flashcard quality**   | High — scripture references, key terms, doctrinal concepts map perfectly to term/definition pairs | Very high — art history is "flashcard-native" with artist/work/period/style pairings[^54][^55] | Medium — some concepts work ("What is Big-O?") but code snippets are awkward on flashcards |
| **Quiz generation**     | Strong — multiple choice on doctrines, dates, scripture context                                   | Very strong — "identify the artist/period/style" questions are natural[^56][^57]               | Mixed — conceptual MCQs work, but coding questions need execution to verify                |
| **Chat/RAG value**      | High — cross-referencing scriptures, explaining doctrinal nuance                                  | High — comparing movements, contextualizing works                                              | Very high — but RAG for code has known limitations[^58][^59]                               |
| **Code-as-tool value**  | Low — minimal computation                                                                         | Low — maybe timeline math                                                                      | Very high — solving algorithms, running examples                                           |
| **Demo impressiveness** | Moderate — niche audience                                                                         | High — visual, universally relatable                                                           | High — technical audience loves seeing code execution                                      |
| **BYU relevance**       | Very high — every BYU student takes religion classes                                              | Moderate — niche major but universally interesting                                             | High — relevant to CS/engineering students                                                 |

### Recommendation: Art History

Art history is the strongest MVP domain for several reasons:

- **Flashcard-native**: Art history is fundamentally about memorizing associations — artist → work → period → style → technique. This maps directly to flashcard term/definition pairs, meaning the study generation agent produces visibly excellent output.[^60][^54]
- **Quiz-native**: "Which artist painted this?" / "What period does this belong to?" / "Compare Baroque and Rococo" are natural multiple-choice and short-answer questions.[^56][^61]
- **Visually impressive demo**: Art history content is inherently visual. A demo where you upload slides full of paintings, then chat about artistic movements, then generate flashcards with artwork references, looks _stunning_ to any audience — not just CS people.
- **Clean ingestion pipeline**: Inputs are primarily PDF slides and textbook chapters. OCR needs are minimal (typed text, not equations). No need for Mathpix or code execution in the MVP — reducing API costs and complexity.
- **Universal appeal on a resume**: An art history study tool demonstrates the AI/agentic engineering skills while being accessible to non-technical interviewers. A hiring manager at a startup can watch your demo and _understand it_ without knowing Python.
- **BYU connection**: Given your interest in Renaissance and Northern Renaissance art history, you can dogfood this with your own coursework.

### What Changes for an Art History MVP

Scoping to art history simplifies the architecture in specific ways:

| Component          | General Version                   | Art History MVP                                                             |
| ------------------ | --------------------------------- | --------------------------------------------------------------------------- |
| OCR pipeline       | Mathpix + Claude Vision + Whisper | Claude Vision only (typed text + image descriptions)                        |
| Code-as-tool       | Full Python sandbox               | Not needed for MVP — add later for CS expansion                             |
| Flashcard format   | Generic term/definition           | Art-specific: Artist, Title, Date, Period, Medium, Style, Significance      |
| Quiz types         | Generic MCQ                       | Art-specific: identification, comparison, timeline ordering, style matching |
| Chat prompts       | Generic study assistant           | Art history specialist with knowledge of periods, movements, techniques     |
| Embedding strategy | Generic chunking                  | Chunk per artwork/concept with rich metadata (period, artist, date)         |

### Art History Chat Agent Specialization

The system prompt for the RAG chat agent gets domain-tuned:

```python
ART_HISTORY_SYSTEM = """
You are an expert art history study assistant with deep knowledge of
Western art from antiquity through contemporary periods.

When answering questions:
1. Always ground your response in the student's uploaded course materials first.
2. Use proper art historical terminology (chiaroscuro, sfumato, tenebrism, etc.)
3. When discussing an artwork, reference: artist, title, date, medium if known.
4. Cite which lecture/reading the information comes from.
"""
```

---

## Your Current Workflow (What This Tool Replaces)

Based on how you use Perplexity to study, your current workflow looks like this:

1. **Paste or upload** your study guide / lecture notes into Perplexity
2. **Give a detailed prompt** describing what you want ("create flashcards for these artworks," "quiz me on these concepts," "explain this topic using my notes")
3. **Iterate** with follow-up prompts to refine, add more, or change difficulty
4. **Copy the output** somewhere useful (Quizlet, your own notes, etc.)

This is essentially what Perplexity's Study Mode does — upload materials, generate flashcards/quizzes, and chat with your content. The key pain points your tool can solve better:[^62][^63]

- **Persistence**: In Perplexity, each thread is ephemeral. You have to re-upload and re-prompt every session. Your tool stores materials permanently per class and accumulates knowledge over time.
- **Structure**: Perplexity returns flashcards as text in a chat thread. Your tool stores them in a database, lets you browse/edit/export them, and generates new ones without losing the old.
- **Class-scoped RAG**: Perplexity Spaces allows file uploads, but your tool embeds everything into a vector store scoped to one class, so the chat agent always has the full context of _all_ your materials, not just what you uploaded in one thread.[^64]
- **Domain tuning**: Your system prompt is art-history-specific by default — it knows to extract artists, periods, movements, and techniques without you having to prompt for it every time.

This framing is important for your CS 301R presentation: you're not building "Perplexity but worse" — you're building a **persistent, class-scoped, domain-tuned study agent** that eliminates the re-prompting and copy-pasting friction.

---

## MVP Scope: What's In, What's Not

The MVP must answer one question: **"Can I upload my art history notes and get useful study materials back?"** Everything else is post-MVP.

### MVP Features (Must-Have)

- **Upload a PDF or text file** → deterministic extraction + chunking + embedding
- **Chat with your materials** → RAG agent with tool use (the core agentic feature)
- **Generate flashcards** → single LLM call with structured output (term/definition pairs)
- **Export flashcards to CSV** → deterministic format conversion for Quizlet/Anki import
- **Basic UI** → one page: upload zone, chat panel, flashcard list with export button

That's it. Five features. One page. One class at a time.

### Post-MVP Features (Add Later)

| Feature                         | Why It’s Post-MVP                                                   |
| ------------------------------- | ------------------------------------------------------------------- |
| Quiz generation                 | Flashcards prove the concept; quizzes are a second output format    |
| Reflection/self-critique loop   | Quality optimization — nice but not needed to demonstrate the agent |
| Code-as-tool sandbox            | Only needed for CS domain expansion                                 |
| Streaming SSE responses         | Improves UX but `fetch` + loading spinner works fine for MVP        |
| Mathpix OCR                     | Only needed for STEM content                                        |
| Whisper transcription           | Only needed for audio inputs                                        |
| Multi-class dashboard           | MVP works with one class at a time                                  |
| Dark mode toggle                | Ship dark-only; toggle is cosmetic                                  |
| Drag-and-drop upload            | A basic file input works fine                                       |
| Railway deployment              | Run locally for the class demo; deploy when you want a live URL     |
| Card flip animation             | CSV export is the MVP output; interactive UI comes later            |
| Domain expansion (Religion, CS) | The whole point of scoping to art history first                     |

---

## MVP Build Order

**Phase 1 — Foundation (Days 1–2)** ✅ COMPLETE

- Copy `tools.py`, `run_agent.py`, `usage.py` from your course repo
- SQLite schema with `inputs` and `flashcards` tables
- ChromaDB collection initialization
- File save to `data/uploads/`
- Gradio MVP interface (3-panel layout)

**Phase 2 — Ingestion Pipeline (Days 3–4)** ✅ COMPLETE

- PDF text extraction with pypdf
- Plain text / DOCX support
- Chunking with `RecursiveCharacterTextSplitter` + embed into ChromaDB via `text-embedding-3-small`
- Integrated into Gradio upload workflow with detailed status feedback
- Full end-to-end pipeline: upload → extract → chunk → embed → store

**Phase 2.5 — Additional UX Improvements (2026-03-20)** ✅ COMPLETE

### File Management
- Individual file deletion with confirmation dialogs
- Bulk "Clear All Files" functionality for testing
- Three-layer cleanup (disk, SQLite, ChromaDB)
- Cascade deletion for related flashcards
- Real-time file list updates

### Class Switching
- Dropdown selector for existing classes
- "Create New Class" conditional UI
- Auto-updating dropdown on class creation
- Hidden state management for clean event handling
- Context switching with file list updates

**Phase 3 — RAG Chat Agent (Days 5–7)** ✅ COMPLETE

- ✅ `search_class_materials` tool registered in ToolBox
- ✅ Agent loop using course `run_agent()` pattern with OpenAI GPT-4o-mini
- ✅ Art history expert system prompt with tool usage guidelines
- ✅ Wired into Gradio `gr.Chatbot` component
- ✅ Chat history persistence (in-session via Gradio state)
- ✅ **Spelling correction** with rapidfuzz (65% threshold, catches "Alderfini" → "Arnolfini")
- ✅ **Web search tool** via Tavily API for historical context lookup
- ✅ **Single-line chat input** (Enter to send, standard chat UX)

**Phase 3.5 — Chat Response Optimization (Optional)** 🚧 TODO

- Limit response length to reduce verbosity and costs
- Options: system prompt instructions, max_tokens parameter, or dynamic limits
- Trade-off: faster/cheaper vs. detailed art history explanations
- Recommended: Start with system prompt "Keep responses concise (2-3 paragraphs max)"

**Phase 4 — Flashcard Generation + Export (Days 8–9)** ✅ COMPLETE

- ✅ Structured Output call with JSON schema for `{term, definition}` pairs
- ✅ Store flashcards in SQLite
- ✅ CSV export function for Quizlet/Anki (`app/pipelines/exporters.py`)
- ✅ UI display with topic input and count slider
- ✅ Agentic RAG workflow: agent searches materials then generates structured flashcards
- ✅ Intent parsing for scope guidance (terms vs. people vs. artworks)
- ✅ Post-generation deduplication
- **Deviation**: Used OpenAI Structured Outputs (Responses API) instead of Anthropic. Gradio `gr.Dataframe` replaced by Flask table with inline editing.

**Phase 5 — Flask Migration & UI Polish** ✅ COMPLETE

- ✅ Migrated from Gradio to Flask + Tailwind CSS + HTMX (Gradio's limitations caused 4 rounds of UI fixes before switching)
- ✅ App factory pattern with Blueprints (`main_bp` + `api_bp`)
- ✅ ~22 REST API endpoints in `app/routes/api.py`
- ✅ HTMX partial rendering for dynamic tab content
- ✅ Flashcard sets with auto-creation per generation
- ✅ Per-card inline editing (edit/delete individual flashcards)
- ✅ Custom accessible dropdowns with keyboard navigation (class selector, set selector)
- ✅ Dark mode with localStorage persistence
- ✅ Toast notifications and modal confirmations (6 `<dialog>` elements)
- ✅ Chat markdown rendering (marked.js + DOMPurify)
- ✅ File management with cascade deletion (disk + SQLite + ChromaDB)
- ✅ Custom migration system (`app/migrations/`)
- **Deviation**: Original Phase 5 was "Polish & Testing" for the Gradio MVP. Flask migration was originally post-MVP but was pulled forward when Gradio proved too limiting for production UX.

---

## Post-MVP Expansion Roadmap

| Phase    | What to Add                                  | Effort   | Status |
| -------- | -------------------------------------------- | -------- | ------ |
| **v0.2** | Quiz generation + interactive quiz UI        | 3–4 days | TODO   |
| **v0.3** | Streaming SSE for chat responses             | 1–2 days | TODO   |
| **v0.4** | Reflection loop on flashcard/quiz quality    | 2–3 days | TODO   |
| **v0.5** | Multi-class dashboard, class CRUD            | 2–3 days | ✅ DONE (class switching, CRUD, file management) |
| **v0.6** | Railway deployment + custom domain           | 1 day    | TODO   |
| **v0.7** | Whisper transcription for audio inputs       | 1–2 days | TODO   |
| **v0.8** | Mathpix OCR + code-as-tool for CS domain     | 3–4 days | TODO (OCR pipeline exists but disabled) |
| **v0.9** | Religion domain profile                      | 2–3 days | TODO   |
| **v1.0** | Polished README, demo video, GitHub showcase | 2–3 days | TODO   |

Additional completed items not in original roadmap:
- ✅ Dark mode toggle (originally listed as post-MVP cosmetic)
- ✅ Flask migration (originally post-MVP, pulled forward)
- ✅ Flashcard sets with per-set management
- ✅ Spelling correction via rapidfuzz
- ✅ Web search tool via Tavily API
- ✅ Per-card inline editing

---

## Deviations from Original Plan

This section documents where the implementation diverged from the original design, added during the Flask migration (April 2026).

### 1. Frontend: Gradio → Flask (Pulled Forward)

**Plan**: Gradio Blocks for MVP, Flask + Tailwind post-MVP.
**Actual**: Gradio was used through Phase 4 but its limitations (no custom styling, broken modal dialogs, poor chat UX, no dark mode control) caused 4 rounds of UI fixes. Migrated to Flask + Tailwind CSS + HTMX during Phase 5 instead of post-MVP.

### 2. LLM Provider: Anthropic → OpenAI for Agent Loops

**Plan**: Claude API (Anthropic) for tool use agent loops.
**Actual**: OpenAI GPT-4o-mini with Responses API for both chat agent and flashcard generation. Anthropic SDK remains a dependency but is not used for the main agent loops. OpenAI was chosen for Structured Outputs support and the Responses API's tool calling ergonomics.

### 3. Embeddings: Local → API

**Plan**: `all-MiniLM-L6-v2` (local, free).
**Actual**: `text-embedding-3-small` (OpenAI API, 1536 dimensions). Higher quality embeddings justified the small per-query cost for an art history RAG use case.

### 4. Route Structure: Per-Resource → Consolidated

**Plan**: Separate blueprint files per resource (`classes.py`, `inputs.py`, `chat.py`, `flashcards.py`, `quizzes.py`).
**Actual**: Two files — `main.py` (page routes + HTMX partials) and `api.py` (~22 REST endpoints). Simpler for the current project size.

### 5. Features Not in Original Plan

| Feature | Why Added |
|---------|-----------|
| **Flashcard sets** | Organize flashcards by generation; auto-named from topic |
| **Spelling correction** (rapidfuzz, 65% threshold) | Art history names are hard to spell ("Alderfini" → "Arnolfini") |
| **Web search** (Tavily API) | Historical context beyond uploaded course materials |
| **Custom migration system** | Lightweight alternative to Alembic for schema changes |
| **Per-card inline editing** | Users needed to fix individual flashcard errors |
| **Custom accessible dropdowns** | Native `<select>` couldn't be styled for dark mode |

### 6. Features Still TODO from Original Plan

| Feature | Original Phase | Notes |
|---------|---------------|-------|
| Quiz generation | v0.2 | `quizzes` table exists, no agent implementation |
| Streaming SSE | v0.3 | Chat uses fetch + typing indicator instead |
| Reflection loop | v0.4 | Flashcard quality relies on prompt engineering only |
| Artwork image fetching | Phase 4.5 | Schema has `image_url` column, API code in INITIAL_PLAN.md |
| Whisper transcription | v0.7 | Pipeline stub exists, not wired up |
| Response length limiting | Phase 3.5 | Documented as future enhancement |
| Deployment | v0.6 | Running locally only |

---

## Phase 6: RAG System Overhaul (Completed)

The initial RAG implementation used flat, section-unaware chunking. Documents like structured study guides (with repeating sections and terms lists) lost their structure when chunked, causing the agent to return incomplete results and hallucinate terms from general knowledge.

### Problems Identified

1. **Flat chunking lost document structure** — 800-char chunks with no section awareness. A chunk containing "Philip the Bold, Flanders, Chartreuse de Champmol..." had no link to "Early Northern Renaissance."
2. **Single-mode search tool** — Only semantic similarity search. No way to filter by section or keyword. The agent couldn't target specific document sections.
3. **Spelling correction mangled normal English** — 65% fuzzy match threshold turned "table" → "Marble" (73% match) and "format" → "Reformation" (71% match).
4. **Agent hallucinated terms** — System prompt told agent to "use proper art historical terminology (chiaroscuro, sfumato, tenebrism, impasto)" which encouraged injecting general knowledge.
5. **ChatMessage.to_dict() included `created_at`** — Broke OpenAI Responses API on 2nd+ messages with "Unknown parameter" error.
6. **Export files orphaned on disk** — Written to `data/uploads/exports/` and never cleaned up.

### Solutions Implemented

1. **Section-aware chunking** (`app/pipelines/section_detector.py`):
   - Detects section headers (title-case, ALL-CAPS, markdown `#`)
   - Strips parenthetical instructor notes before detection
   - Compound names for subsections: `"Early Northern Renaissance > Terms, People, and Places to Know"`
   - Chunk text prefixed with `[Section: ...]` for embedding disambiguation
   - Section stored as ChromaDB metadata

2. **Upgraded search tool** (`app/agents/chat_agent.py`):
   - Three search modes: semantic (query), section filter (metadata), keyword filter (document text)
   - New `list_sections` tool for discovering available sections
   - Agent calls `list_sections` first, then searches per-section for comprehensive coverage

3. **Spelling correction hardened**: Added 200+ common English words exclusion set. Only non-common words are candidates for fuzzy correction.

4. **Anti-hallucination system prompt**: Removed hallucination-encouraging instructions. Added strong grounding constraint: "ONLY include information found in search results."

5. **Source attribution**: Every response ends with grouped "Sources:" section listing which documents/sections were used.

6. **Export from memory**: `io.BytesIO` instead of temp files. No orphaned exports.

7. **ChatMessage.to_dict() fix**: Only returns `role` and `content`.

8. **Flashcard limits**: Increased from 15 to 60 (config, agent, UI slider).

9. **Cleanup**: Deleted legacy `gradio_app.py`. Added `reset_db.py` script.

---

## References

1. [Agentic AI Systems More Expensive Than LLMs Due to Inference ...](https://www.linkedin.com/posts/prince-goyal_agenticsystems-agenticai-llmcostmodel-activity-7410083623565164544-Jgxg) - Agentic AI systems are structurally more expensive than traditional LLM models - and this is an arch...

2. [Building Effective AI Agents - Anthropic](https://www.anthropic.com/engineering/building-effective-agents)

3. [Building Effective AI Agents](https://www.anthropic.com/engineering/building-effective-agents?subjects=claude) - Discover how Anthropic approaches the development of reliable AI agents. Learn about our research on...

4. [AI Workflows vs. AI Agents vs. Everything in between](https://blog.tobiaszwingmann.com/p/ai-workflows-vs-ai-agents-vs-everything-in-between) - When Not to Use It · There's no stable sequence of predefined steps · Inputs vary significantly or r...

5. [Deciding when not to use AI agents | Edgar Muyale posted on the topic](https://www.linkedin.com/posts/edgar-muyale-502a56248_aiagents-activity-7421841213860990976-z_C3) - My rule for deciding when not to use an AI agent Not every problem needs an agent. If a task is dete...

6. [AI Workflows vs. AI Agents](https://www.promptingguide.ai/agents/ai-workflows-vs-ai-agents) - A Comprehensive Overview of Prompt Engineering

7. [Deterministic vs Fuzzy Workflows: When to Use AI Agents](https://www.linkedin.com/posts/ddewinter_in-case-you-missed-the-memo-you-should-not-activity-7401681578554408960-Gs9J) - In case you missed the memo, you SHOULD NOT build an AI agent to execute a workflow when... 🔷 The wo...

8. [Introduction | Mathpix Docs](https://docs.mathpix.com) - Mathpix OCR recognizes printed and handwritten STEM document content, including math, text, tables, ...

9. [How to Use OpenAI's Whisper for Perfect Transcriptions (Speech to ...](https://www.youtube.com/watch?v=dg_TWk8Zfjk) - In this step-by-step tutorial, I show you how to use OpenAI's Whisper AI to get incredibly accurate ...

10. [Chunking Strategies for RAG: Best Practices and Key Methods](https://unstructured.io/blog/chunking-for-rag-best-practices) - Chunking strategies for RAG directly affect retrieval precision and LLM response quality. Compare fi...

11. [Creating sets by importing content - Quizlet Help Center](https://help.quizlet.com/hc/en-us/articles/360029977151-Creating-sets-by-importing-content) - Quickly create new flashcard sets based on existing notes or documents by importing them right into ...

12. [Anthropic's LLM Workflows vs Agents | Nitin Monga posted on the topic](https://www.linkedin.com/posts/nitinmonga-ai_ai-agenticai-anthropic-activity-7425870539715538944-05Ti) - 𝗪𝗼𝗿𝗸𝗳𝗹𝗼𝘄𝘀 𝘃𝘀. 𝗔𝗴𝗲𝗻𝘁𝘀 Anthropic categorizes both as 𝗮𝗴𝗲𝗻𝘁𝗶𝗰 𝘀𝘆𝘀𝘁𝗲𝗺𝘀, but draws an important architect...

13. [AI Agents vs. Deterministic Workflows - LinkedIn](https://www.linkedin.com/pulse/ai-agents-vs-deterministic-workflows-chiheb-dkhil-vlsve) - A Twelve-Dimension Enterprise Decision Framework Beyond the Hype: Engineering Discipline for the Age...

14. [When Less Reasoning Is More: Designing Efficient Agentic Workflows](https://www.linkedin.com/pulse/when-less-reasoning-more-designing-efficient-agentic-workflows-aj3qe) - Not Everything Needs to Be an Agent: How to achieve a trade-off between deterministic and reasoning ...

15. [Deterministic vs Agentic Workflows: How to Choose What to ...](https://www.nexaforge.dev/blog/ai-101/deterministic-vs-agentic-workflows) - A practical decision framework for business leaders: when to use AI agents versus traditional determ...

16. [Mathpix now supports text OCR](https://mathpix.com/blog/mathpix-text-ocr) - Mathpix now reads a lot more than just math.

17. [OpenAI Whisper - Converting Speech to Text - GeeksforGeeks](https://www.geeksforgeeks.org/artificial-intelligence/openai-whisper-converting-speech-to-text/) - Speech Recognition: Whisper enables the conversion of audio recordings into written text. This funct...

18. [Step 5: Query The Model](https://www.gettingstarted.ai/tutorial-chroma-db-best-vector-database-for-langchain-store-embeddings/) - The LangChain framework allows you to build a RAG app easily. In this tutorial, see how you can pair...

19. [Learn How to Use Chroma DB: A Step-by-Step Guide | DataCamp](https://www.datacamp.com/tutorial/chromadb-tutorial-step-by-step-guide) - Learn how to use Chroma DB to store and manage large text datasets, convert unstructured text into n...

20. [Chunking Strategies to Improve Your RAG Performance](https://weaviate.io/blog/chunking-strategies-for-rag) - Learn how chunking strategies improve LLM RAG pipelines, retrieval quality, and agent memory perform...

21. [Method to Import CSV Files into AnkiMobile (Without a Computer) – From GPT to Anki](https://www.reddit.com/r/Anki/comments/1j81atw/method_to_import_csv_files_into_ankimobile/) - Method to Import CSV Files into AnkiMobile (Without a Computer) – From GPT to Anki

22. [Import CSV files to Anki to Automatically Create Flashcards](https://bzzz.hashnode.dev/anki-hack-automatically-create-customized-cards-by-importing-from-a-spreadsheet) - Step-by-step tutorial on how to import CSV files to Anki to automatically create flashcards

23. [What is an LLM Code Interpreter? Benefits & How it Works - Iguaziowww.iguazio.com › glossary › what-is-llm-code-interpreter](https://www.iguazio.com/glossary/what-is-llm-code-interpreter/) - Learn what an LLM Code Interpreter is, how it works, and discover its key benefits for software deve...

24. [Sandboxes - Docs by LangChain](https://docs.langchain.com/oss/python/deepagents/sandboxes) - Execute code in isolated environments with sandbox backends

25. [Code Execution in AI Agents: Safe and Effective Patterns | Michael ...](https://michaeljohnpena.com/blog/2024-05-26-code-execution-ai-agents/) - Code Execution in AI Agents: Safe and Effective Patterns - A blog post by Michael John Peña

26. [LangChain Sandbox: Run Untrusted Python Safely for AI Agents](https://www.youtube.com/watch?v=FBnER2sxt0w) - 🛡️ Introducing LangChain Sandbox: run untrusted Python safely in your AI agents Powered by Pyodide ...

27. [Anthropic thinks you should build agents like this - AI Hero](https://www.aihero.dev/building-effective-agents) - Learn how agents differ from workflows and why to start AI projects with LLM APIs, not frameworks.

28. [Structured model outputs | OpenAI API](https://developers.openai.com/api/docs/guides/structured-outputs/) - Understand how to ensure model responses follow specific JSON Schema you define.

29. [Constrained Decoding](https://openai.com/index/introducing-structured-outputs-in-the-api/) - We are introducing Structured Outputs in the API—model outputs now reliably adhere to developer-supp...

30. [CLAUDE.md](uloaded to perplexity) - # CLAUDE.md This file provides guidance to Claude Code (claude.ai/code) when working with code in t...

31. [Design Patterns for Effective AI Agents - AI Changes Everything](https://patmcguinness.substack.com/p/design-patterns-for-effective-ai) - Anthropic gives advice on building AI Agents. Simple workflows connecting capable tools and advanced...

32. [SQLite vs. Chroma: A Comparative Analysis for Managing Vector ...](https://dev.to/stephenc222/sqlite-vs-chroma-a-comparative-analysis-for-managing-vector-embeddings-4i76) - Navigate through a comparison of SQLite, boosted with the `sqlite-vss` extension, and Chroma for man...

33. [How To Structure a Large Flask Application with Flask Blueprints ...](https://www.digitalocean.com/community/tutorials/how-to-structure-a-large-flask-application-with-flask-blueprints-and-flask-sqlalchemy) - In this tutorial, you'll use Flask blueprints to structure a web application with three components: ...

34. [How To Structure a Large Flask Application-Best Practices for 2025](https://dev.to/gajanan0707/how-to-structure-a-large-flask-application-best-practices-for-2025-9j2) - A well-structurally designed Flask RESTful API is readable, maintainable, scalable as well as ease o...

35. [4 Agentic AI Design Patterns to Build Intelligent Workflows - ProjectPro](https://www.projectpro.io/article/agentic-ai-design-patterns/1126) - Explore how the 4 key agentic AI design patterns like Planning, Reflection, Tool Use, and Multi-Agen...

36. [7 Must-Know Agentic AI Design Patterns - Machine Learning Mastery](https://machinelearningmastery.com/7-must-know-agentic-ai-design-patterns/) - In this article, you will learn seven proven agentic AI design patterns, when to use each, and how t...

37. [Implementing the ReAct LLM Agent pattern the hard way and the easy way](https://jschrier.github.io/blog/2024/01/13/Implementing-the-ReAct-LLM-Agent-pattern-the-hard-way-and-the-easy-way.html) - ReAct prompting is a way to have large language models (LLM) combine reasoning traces and task-speci...

38. [How to Build Enterprise Grade AI Agents with Agentic Design Patterns](https://www.tungstenautomation.de/learn/blog/build-enterprise-grade-ai-agents-agentic-design-patterns) - A practical guide to agentic AI design patterns and how they enable reliable, controllable, and ente...

39. [Production LLM: Improving Agent Quality Through Self ...](https://alexostrovskyy.com/production-llm-improving-agent-quality-through-self-reflection/) - Artificial Intelligence (AI) agents are becoming increasingly capable, automating tasks, answering q...

40. [AI Agent Reflection and Self-Evaluation Patterns | Zylos Research](https://zylos.ai/research/2026-03-06-ai-agent-reflection-self-evaluation-patterns) - A deep dive into reflection, self-critique, and verification patterns that enable AI agents to asses...

41. [Add AI Projects that solve Real...](https://www.projectpro.io/article/artificial-intelligence-portfolio/1140) - Learn to build an Artificial Intelligence portfolio that speaks for you by picking impactful project...

42. [ML Engineer Portfolio Projects That Will Get You Hired in 2025](http://www.interviewnode.com/post/ml-engineer-portfolio-projects-that-will-get-you-hired-in-2025) - Introduction: Why Portfolios Matter More Than Ever in 2025If you’re aiming to land a machine learnin...

43. [10 GitHub README Examples That Get Stars: A Developer's Guide to ...](https://blog.beautifulmarkdown.com/10-github-readme-examples-that-get-stars) - Discover the secrets behind GitHub READMEs that get thousands of stars. Learn from real examples, be...

44. [Building and Styling UI Components Using Tailwind CSS - Tailgrids](https://tailgrids.com/blog/component-styling-and-building-components) - Learn how to build and style UI components effortlessly with Tailwind CSS. Discover tips for creatin...

45. [Implementing HTMX with Flask and Jinja2 for Dynamic ...](https://dev.to/hexshift/implementing-htmx-with-flask-and-jinja2-for-dynamic-content-rendering-2bck) - HTMX is a lightweight JavaScript library that allows you to build dynamic, modern web applications.....

46. [Dark mode - Core concepts - Tailwind CSS](https://tailwindcss.com/docs/dark-mode) - Using variants to style your site in dark mode.

47. [How To Implement Instant Search with Flask and HTMX](https://www.freecodecamp.org/news/how-to-implement-instant-search-with-flask-and-htmx/) - Instant search is a feature that shows search results as users type their query. Instead of waiting ...

48. [Using htmx with Flask](https://app.studyraid.com/en/read/1955/32839/using-htmx-with-flask) - Flask, a popular Python web framework, pairs exceptionally well with htmx to create dynamic and inte...

49. [Dynamic Forms Handling with HTMX and Python Flask](https://www.geeksforgeeks.org/python/dynamic-forms-handling-with-htmx-and-python-flask/) - Your All-in-One Learning Portal: GeeksforGeeks is a comprehensive educational platform that empowers...

50. [Gradio vs Streamlit vs Dash vs Flask](https://towardsdatascience.com/gradio-vs-streamlit-vs-dash-vs-flask-d3defb1209a2/) - Comparing several web UI tools for data science!

51. [Gradio vs Streamlit vs Dash vs Flask - Choosing the Right Tool](https://howik.com/gradio-vs-streamlit-vs-dash-vs-flask) - discover the best tool for your web app needs with our in-depth comparison of gradio vs streamlit vs...

52. [Deploy a Flask App | Railway Guides](https://docs.railway.com/guides/flask) - Learn how to deploy a Flask app to Railway with this step-by-step guide. It covers quick setup, one-...

53. [Deploy Flask SQLite CRUD - Railway](https://railway.com/deploy/flask-sqlite-crud) - Deploy Flask SQLite CRUD to the cloud for free with Railway, the all-in-one intelligent cloud provid...

54. [Why Art History Is Perfect...](https://www.seegreatart.art/how-to-master-art-history-ai-flashcards-to-enhance-your-learning/) - Art history, with its long list of names, styles, and dates, often needs strong memory skills. AI fl...

55. [Instant AP Art History Flashcards with AI](https://www.cogniguide.app/flashcards/flashcards-for-ap-art-history) - Generate expert AP Art History flashcards instantly. Upload lecture slides or PDF notes, and let our...

56. [Instant AI Art Quiz Generator & Answers](https://www.cogniguide.app/quizzes/art-quiz-with-answers) - Generate custom art quizzes with answers instantly. Upload art history PDFs or prompt the AI to crea...

57. [Instant AI Art Quiz Generator | Custom Visual Tests - CogniGuide](https://www.cogniguide.app/quizzes/art-quiz-questions-with-pictures) - Generate art quiz questions with pictures instantly by uploading your lecture slides or images. Our ...

58. [Why I Stopped Using RAG for Coding Agents (And You Should ...](https://jxnl.co/writing/2025/09/11/why-i-stopped-using-rag-for-coding-agents-and-you-should-too/) - Why leading coding agent companies are abandoning embedding-based RAG in favor of direct, agentic ap...

59. [Why RAG Pipelines Fail at Modernizing Legacy Codebases](https://www.codeant.ai/blogs/why-rag-fails-legacy-codebases) - If you've ever worked in a codebase that predates the current decade, you know the feeling. Function...

60. [Generate AI History of Art Flashcards Instantly - CogniGuide](https://www.cogniguide.app/flashcards/history-of-art-flashcards) - Need history of art flashcards? Create your own AI flashcards in minutes from notes or prompts, util...

61. [Instant Easy Art Quiz Generator | AI Practice - CogniGuide](https://www.cogniguide.app/quizzes/easy-art-quiz-questions-and-answers) - Upload your art history slides, images, or notes, and watch our AI instantly structure precise quizz...

62. [What is Learn Mode? | Perplexity Help Center](https://www.perplexity.ai/help-center/en/articles/12120542-what-is-learn-mode) - In Learn Mode, upload any course materials, readings, study guides, or lecture notes and ask Perplex...

63. [What We Shipped - September 5th - Perplexity Changelog](https://www.perplexity.ai/changelog/what-we-shipped-september-5th) - What We Shipped - September 5th

64. [A student's guide to using Perplexity Spaces](https://www.perplexity.ai/hub/blog/a-student-s-guide-to-using-perplexity-spaces) - Explore Perplexity's blog for articles, announcements, product updates, and tips to optimize your ex...
