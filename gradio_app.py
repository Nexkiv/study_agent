"""
Gradio MVP for StudyAgent.

Three-panel layout:
1. Upload panel: File upload and class management
2. Chat panel: RAG agent conversation
3. Flashcard panel: Generated cards and export

Run:
    python gradio_app.py
"""
import gradio as gr
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from app.config import validate_config, ensure_directories, UPLOAD_PATH
from app.extensions import db, get_chroma_client
from app.models import Class, Input, Flashcard, ChatMessage
from app.utils.file_handler import save_upload, delete_upload
from app.pipelines.ingestion import process_upload, extract_pdf
from app.pipelines.chunking import chunk_text, generate_embeddings, delete_embeddings
from app.agents.run_agent import run_agent
from app.agents.tools import ToolBox
from app.agents.chat_agent import (
    get_async_openai_client,
    create_rag_agent_config,
    create_search_tool,
    execute_python,
    correct_spelling,
    search_web
)
from app.agents.study_agent import generate_flashcards_for_topic
from app.pipelines.exporters import export_to_quizlet, export_to_anki
import asyncio
from flask import Flask

app = None

def init_app():
    """Initialize Flask app and database."""
    global app

    validate_config()
    ensure_directories()

    from app.config import SQLALCHEMY_DATABASE_URI, SQLALCHEMY_TRACK_MODIFICATIONS
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = SQLALCHEMY_DATABASE_URI
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = SQLALCHEMY_TRACK_MODIFICATIONS

    db.init_app(app)
    get_chroma_client()

    # Run database migrations automatically on startup
    from app.migrations import MigrationRunner
    from app.config import PROJECT_ROOT, DATABASE_PATH
    migrations_dir = PROJECT_ROOT / 'app' / 'migrations'
    runner = MigrationRunner(DATABASE_PATH, migrations_dir)

    try:
        runner.run_migrations()
    except Exception as e:
        print(f"⚠️ Migration failed: {e}")
        print("Database may be in inconsistent state. Run: python init_db.py")
        raise

    print("✓ StudyAgent initialized")

def upload_file(file, class_name, input_name, input_type):
    """Handle file upload with full ingestion pipeline (Phase 2)."""
    if file is None:
        return "⚠️ Please select a file", None

    if not class_name:
        return "⚠️ Please enter a class name", None

    if not input_name:
        return "⚠️ Please name this input", None

    try:
        with app.app_context():
            # Get or create class
            class_obj = Class.query.filter_by(name=class_name).first()
            if not class_obj:
                class_obj = Class(name=class_name)
                db.session.add(class_obj)
                db.session.commit()

            # Extract original filename
            original_filename = Path(file.name).name if hasattr(file, 'name') else 'upload.pdf'

            # Save file to disk
            relative_path = save_upload(file, class_obj.id, original_filename)

            # Get absolute path for processing
            absolute_path = UPLOAD_PATH / relative_path

            # Phase 2: Extract text from uploaded file
            try:
                ext = Path(absolute_path).suffix.lower()

                if ext == '.pdf':
                    raw_text, page_count, needs_ocr = extract_pdf(str(absolute_path))
                    extraction_method = 'pypdf'
                elif ext == '.pptx':
                    from app.pipelines.ocr import extract_pptx
                    raw_text = extract_pptx(str(absolute_path))
                    page_count = 1
                    needs_ocr = False
                    extraction_method = 'pptx'
                else:
                    raw_text = process_upload(str(absolute_path), class_obj.id, input_name, input_type)
                    page_count = 1
                    needs_ocr = False
                    extraction_method = 'standard'
            except Exception as e:
                return f"❌ Text extraction failed: {str(e)}", None

            # Phase 2: Chunk and embed the text
            try:
                chunks = chunk_text(raw_text)
                generate_embeddings(class_obj.id, input_name, chunks)
            except Exception as e:
                return f"❌ Embedding generation failed: {str(e)}", None

            # Create input record with extracted text
            input_obj = Input(
                class_id=class_obj.id,
                name=input_name,
                input_type=input_type,
                file_path=relative_path,
                raw_text=raw_text,
                extraction_method=extraction_method
            )
            db.session.add(input_obj)
            db.session.commit()

            # Get updated file list
            inputs = Input.query.filter_by(class_id=class_obj.id).all()
            file_list = [[inp.name, inp.input_type, "✅ Ingested"]
                        for inp in inputs]

            chunk_count = len(chunks)
            status_msg = (
                f"✅ Uploaded: {input_name}\n"
                f"📄 Extracted {len(raw_text):,} characters\n"
                f"🔢 Created {chunk_count} chunks\n"
                f"💾 Stored embeddings in ChromaDB"
            )

            if needs_ocr:
                status_msg += "\n\n⚠️ Low extraction quality detected. Try 'Retry with OCR' below."

            return status_msg, file_list

    except Exception as e:
        return f"❌ Error: {str(e)}", None

def retry_with_ocr(input_id, ocr_method, class_name):
    """
    Re-process a file using OCR.

    Args:
        input_id: ID of input to retry (from dropdown)
        ocr_method: 'tesseract', 'mathpix', or 'claude'
        class_name: Current class name

    Returns:
        (status_message, updated_file_list)
    """
    if not input_id:
        return "⚠️ Please select a file to retry", None

    try:
        with app.app_context():
            input_obj = Input.query.get(input_id)
            if not input_obj:
                return "❌ File not found", None

            file_path = UPLOAD_PATH / input_obj.file_path

            # Import OCR functions
            from app.pipelines.ocr import (
                extract_pdf_with_tesseract,
                extract_pdf_with_mathpix,
                extract_pdf_with_claude_vision
            )

            # Select OCR method
            try:
                if ocr_method == 'tesseract':
                    raw_text = extract_pdf_with_tesseract(str(file_path))
                elif ocr_method == 'mathpix':
                    raw_text = extract_pdf_with_mathpix(str(file_path))
                elif ocr_method == 'claude':
                    raw_text = extract_pdf_with_claude_vision(str(file_path))
                else:
                    return f"❌ Unknown OCR method: {ocr_method}", None
            except RuntimeError as e:
                # Feature disabled or dependencies missing
                return f"❌ {str(e)}", None
            except Exception as e:
                return f"❌ OCR failed: {str(e)}", None

            # Re-chunk and re-embed
            delete_embeddings(input_obj.class_id, input_obj.name)
            chunks = chunk_text(raw_text)
            generate_embeddings(input_obj.class_id, input_obj.name, chunks)

            # Update database
            input_obj.raw_text = raw_text
            input_obj.extraction_method = ocr_method
            db.session.commit()

            file_list = get_current_file_list(class_name)

            return (
                f"✅ Re-extracted with {ocr_method.capitalize()} OCR\n"
                f"📄 Extracted {len(raw_text):,} characters\n"
                f"🔢 Created {len(chunks)} chunks\n"
                f"💾 Updated embeddings"
            ), file_list

    except Exception as e:
        return f"❌ Error: {str(e)}", None

def get_current_file_list(class_name: str) -> list:
    """Get file list for Dataframe display: [[name, type, status], ...]"""
    if not class_name:
        return []

    with app.app_context():
        class_obj = Class.query.filter_by(name=class_name).first()
        if not class_obj:
            return []

        inputs = Input.query.filter_by(class_id=class_obj.id).all()
        return [[inp.name, inp.input_type, "✅ Ingested"] for inp in inputs]

def get_file_choices(class_name: str) -> list:
    """Get dropdown choices: [("Lecture 1 (slides)", 1), ...]"""
    if not class_name:
        return []

    with app.app_context():
        class_obj = Class.query.filter_by(name=class_name).first()
        if not class_obj:
            return []

        inputs = Input.query.filter_by(class_id=class_obj.id).all()
        return [(f"{inp.name} ({inp.input_type})", inp.id) for inp in inputs]

def delete_file(input_id: int, class_name: str) -> tuple[str, list]:
    """Delete a single file and related data. Returns (status_msg, updated_file_list)"""
    try:
        with app.app_context():
            # 1. Query input to get metadata
            input_obj = Input.query.get(input_id)
            if not input_obj:
                return "❌ Error: File not found", get_current_file_list(class_name)

            input_name = input_obj.name
            file_path = input_obj.file_path
            class_id = input_obj.class_id

            # 2. Delete database records (transaction)
            try:
                Flashcard.query.filter_by(input_id=input_id).delete()
                db.session.delete(input_obj)
                db.session.commit()
            except Exception as e:
                db.session.rollback()
                return f"❌ Database error: {str(e)}", get_current_file_list(class_name)

            # 3. Delete ChromaDB chunks (best-effort)
            try:
                delete_embeddings(class_id, input_name)
            except Exception as e:
                print(f"⚠️ ChromaDB warning: {e}")

            # 4. Delete disk file (best-effort)
            if file_path:
                try:
                    delete_upload(file_path)
                except Exception as e:
                    print(f"⚠️ Disk warning: {e}")

            return f"✅ Deleted: {input_name}", get_current_file_list(class_name)

    except Exception as e:
        return f"❌ Error: {str(e)}", get_current_file_list(class_name)

def clear_all_files(class_name: str) -> tuple[str, list]:
    """Delete ALL files for a class. Returns (status_msg, empty_list)"""
    try:
        with app.app_context():
            class_obj = Class.query.filter_by(name=class_name).first()
            if not class_obj:
                return "⚠️ Class not found", []

            inputs = Input.query.filter_by(class_id=class_obj.id).all()
            if not inputs:
                return "⚠️ No files to delete", []

            count = len(inputs)
            class_id = class_obj.id
            file_data = [(inp.id, inp.name, inp.file_path) for inp in inputs]

            # Delete database records (transaction)
            try:
                for input_id, _, _ in file_data:
                    Flashcard.query.filter_by(input_id=input_id).delete()
                Input.query.filter_by(class_id=class_id).delete()
                db.session.commit()
            except Exception as e:
                db.session.rollback()
                return f"❌ Database error: {str(e)}", get_current_file_list(class_name)

            # Best-effort cleanup
            for _, input_name, _ in file_data:
                try:
                    delete_embeddings(class_id, input_name)
                except:
                    pass

            for _, _, file_path in file_data:
                if file_path:
                    try:
                        delete_upload(file_path)
                    except:
                        pass

            return f"✅ Cleared {count} file(s)", []

    except Exception as e:
        return f"❌ Error: {str(e)}", get_current_file_list(class_name)

def get_all_classes() -> list:
    """
    Get all existing classes for dropdown choices.
    Returns list of class names sorted alphabetically, plus 'Create New Class...' option.
    """
    with app.app_context():
        classes = Class.query.order_by(Class.name).all()
        class_names = [cls.name for cls in classes]
        # Add special option at the end
        return class_names + ["➕ Create New Class..."]

def handle_class_selection(selected_value: str):
    """
    Handle class selection from dropdown.
    - If existing class: hide new class input, return selected name
    - If "Create New Class...": show new class input and create button

    Returns: (class_name, new_class_visible, create_btn_visible, file_list, file_selector, ocr_file_selector)
    """
    if selected_value == "➕ Create New Class...":
        return (
            None,  # class_name (hidden state)
            gr.update(visible=True),  # new_class_input
            gr.update(visible=True),  # create_class_btn
            [],  # file_list (empty)
            gr.update(choices=[], value=None),  # file_selector (empty)
            gr.update(choices=[], value=None)  # ocr_file_selector (empty)
        )
    else:
        # Existing class selected
        file_choices = get_file_choices(selected_value)
        return (
            selected_value,  # class_name
            gr.update(visible=False),  # new_class_input
            gr.update(visible=False),  # create_class_btn
            get_current_file_list(selected_value),  # file_list
            gr.update(choices=file_choices, value=None),  # file_selector
            gr.update(choices=file_choices, value=None)  # ocr_file_selector
        )

def create_new_class(new_class_name: str):
    """
    Create a new class and update dropdown choices.

    Returns: (dropdown_update, new_class_visible, create_btn_visible, status)
    """
    if not new_class_name or not new_class_name.strip():
        return (
            gr.update(choices=get_all_classes(), value="➕ Create New Class..."),  # Single update
            gr.update(visible=True),  # Keep input visible
            gr.update(visible=True),  # Keep button visible
            "⚠️ Please enter a class name."  # status
        )

    with app.app_context():
        # Check if class already exists
        existing = Class.query.filter_by(name=new_class_name).first()
        if existing:
            return (
                gr.update(choices=get_all_classes(), value=new_class_name),  # Single update
                gr.update(visible=False),
                gr.update(visible=False),
                f"✅ Switched to existing class: {new_class_name}."
            )

        # Create new class
        new_class = Class(name=new_class_name)
        db.session.add(new_class)
        db.session.commit()

        # Update dropdown choices
        updated_choices = get_all_classes()

        return (
            gr.update(choices=updated_choices, value=new_class_name),  # Single update
            gr.update(visible=False),  # Hide new class input
            gr.update(visible=False),  # Hide create button
            f"✅ Created new class: {new_class_name}."  # status
        )

def delete_class(class_name):
    """
    Delete a class and reset ALL UI state atomically.

    Cascade deletes:
    - All input files (database records + physical files + ChromaDB chunks)
    - All flashcards
    - All chat messages
    - ChromaDB collection

    Returns:
        Tuple of 6 values:
        1. upload_status (str) - Success/error message
        2. class_selector (gr.update) - Refresh dropdown
        3. file_list (list) - Empty list
        4. new_class_input (gr.update) - Hide input
        5. create_class_btn (gr.update) - Hide button
        6. current_class_name (None) - Reset state
    """
    if not class_name or class_name == "➕ Create New Class...":
        return (
            "⚠️ Please select a valid class to delete.",
            gr.update(choices=get_all_classes(), value=None),
            [],
            gr.update(visible=False),
            gr.update(visible=False),
            None
        )

    with app.app_context():
        try:
            # Find class
            class_obj = Class.query.filter_by(name=class_name).first()
            if not class_obj:
                return (
                    f"⚠️ Class '{class_name}' not found",
                    gr.update(choices=get_all_classes(), value=None),
                    [],
                    gr.update(visible=False),
                    gr.update(visible=False),
                    None
                )

            # Delete physical files
            class_upload_dir = UPLOAD_PATH / str(class_obj.id)
            if class_upload_dir.exists():
                import shutil
                shutil.rmtree(class_upload_dir)

            # Delete ChromaDB collection
            try:
                client = get_chroma_client()
                client.delete_collection(f"class_{class_obj.id}")
            except Exception as e:
                print(f"Warning: Could not delete ChromaDB collection: {e}")

            # Delete database records (cascade will handle related records)
            db.session.delete(class_obj)
            db.session.commit()

            # After deletion, reset dropdown and UI state
            updated_choices = get_all_classes()
            return (
                f"✅ Class '{class_name}' deleted successfully.",
                gr.update(choices=updated_choices, value=None),
                [],
                gr.update(visible=False),
                gr.update(visible=False),
                None  # Reset current_class_name to None
            )

        except Exception as e:
            db.session.rollback()
            return (
                f"❌ Error deleting class: {str(e)}",
                gr.update(choices=get_all_classes(), value=None),
                [],
                gr.update(visible=False),
                gr.update(visible=False),
                None
            )

async def chat_with_materials(message, history, class_name):
    """
    RAG chat agent with tool use (Phase 3).

    Uses GPT-4o-mini with search_class_materials and execute_python tools.
    History persists in Gradio state (in-session only).
    """
    # Validation
    if not message or not message.strip():
        return history

    if not class_name:
        return history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "⚠️ Please select a class first"}
        ]

    try:
        # Correct spelling using terms from class materials
        # This helps with misspellings like "Alderfini" -> "Arnolfini"
        with app.app_context():
            corrected_message = correct_spelling(message, class_name)

        # Use corrected message for agent processing
        user_message = corrected_message

        # Initialize ToolBox and register tools
        toolbox = ToolBox()

        # Register search tool (class-specific)
        search_tool = create_search_tool(class_name)
        toolbox.tool(search_tool)

        # Register web search tool
        toolbox.tool(search_web)

        # Register execute_python tool
        toolbox.tool(execute_python)

        # Get OpenAI async client
        client = get_async_openai_client()

        # Create agent config
        agent_config = create_rag_agent_config(class_name)

        # Build clean conversation history for agent
        # Normalize content to plain strings (run_agent adds user message with string content)
        conversation_history = []
        for msg in (history or []):
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                continue

            # Extract text content (handle both string and structured content)
            content = msg["content"]
            if isinstance(content, str):
                text_content = content
            elif isinstance(content, list):
                # Structured content: extract text from all text-type items
                text_content = " ".join(
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict) and "text" in item
                )
            else:
                continue  # Skip malformed messages

            conversation_history.append({
                "role": msg["role"],
                "content": text_content
            })

        # Run agent with fresh history (run_agent mutates history internally)
        with app.app_context():  # Needed for SQLAlchemy queries
            response_text = await run_agent(
                client=client,
                toolbox=toolbox,
                agent=agent_config,
                user_message=user_message,  # Use spelling-corrected message
                history=conversation_history,  # Fresh copy, not Gradio state
                usage=None  # Deferred to Phase 3.5
            )

        # Build final response
        final_response = response_text or "I apologize, but I couldn't generate a response."

        return history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": final_response}
        ]

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        return history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": error_msg}
        ]

def generate_flashcards(class_name, topic):
    """
    Generate flashcards using RAG agent with Structured Outputs.

    Args:
        class_name: Class to generate from
        topic: Optional topic/request (e.g., "Baroque period")

    Returns:
        (status_message, flashcard_dataframe)
    """
    if not class_name:
        return "⚠️ Please select a class", None

    try:
        with app.app_context():
            # Verify class exists
            class_obj = Class.query.filter_by(name=class_name).first()
            if not class_obj:
                return f"⚠️ Class '{class_name}' not found", None

            # Check for uploaded materials
            if not Input.query.filter_by(class_id=class_obj.id).first():
                return "⚠️ No materials uploaded yet. Please upload files first.", None

            # Default topic if not provided
            if not topic or not topic.strip():
                topic = "all major topics covered in the course materials"

            # Call async agent function
            flashcards, agent_status = asyncio.run(
                generate_flashcards_for_topic(class_name, topic, count=15)
            )

            # Check if any flashcards were generated
            if not flashcards:
                return "⚠️ No flashcards generated. Try a more specific topic or check your materials.", None

            # Store flashcards in database
            for card in flashcards:
                flashcard_obj = Flashcard(
                    class_id=class_obj.id,
                    input_id=None,  # Generated from multiple sources
                    term=card['term'],
                    definition=card['definition'],
                    image_url=None  # Phase 4.5
                )
                db.session.add(flashcard_obj)

            db.session.commit()

            # Format for Gradio dataframe
            dataframe_data = [[card['term'], card['definition']] for card in flashcards]

            total_count = Flashcard.query.filter_by(class_id=class_obj.id).count()
            success_msg = f"✅ {len(flashcards)} flashcards generated from {class_name} materials."
            if total_count > len(flashcards):
                success_msg += f" ({total_count} total saved for this class)"

            return success_msg, dataframe_data

    except Exception as e:
        error_str = str(e)
        print(f"❌ Error generating flashcards: {error_str}")
        if "context_length_exceeded" in error_str or "context window" in error_str:
            return "❌ Your materials are too large to process at once. Try a more specific topic, or remove some files and try again.", None
        return "❌ Something went wrong generating flashcards. Please try again.", None

def export_flashcards(class_name, format_choice):
    """
    Export flashcards to CSV/TSV file.

    Args:
        class_name: Class to export from
        format_choice: "Quizlet (TSV)" or "Anki (CSV)"

    Returns:
        File path for Gradio download, or None if no flashcards
    """
    if not class_name:
        return None, "⚠️ Please select a class first."

    try:
        with app.app_context():
            # Get class
            class_obj = Class.query.filter_by(name=class_name).first()
            if not class_obj:
                return None, "⚠️ Class not found."

            # Get all flashcards for this class
            flashcards = Flashcard.query.filter_by(class_id=class_obj.id).all()

            if not flashcards:
                return None, "⚠️ No flashcards to export. Generate flashcards first."

            # Convert to dict format
            cards_data = [
                {"term": card.term, "definition": card.definition}
                for card in flashcards
            ]

            # Export based on format
            if format_choice == "Quizlet (TSV)":
                content = export_to_quizlet(cards_data)
                filename = f"{class_name.replace(' ', '_')}_flashcards_quizlet.tsv"
            else:  # Anki (CSV)
                content = export_to_anki(cards_data)
                filename = f"{class_name.replace(' ', '_')}_flashcards_anki.csv"

            # Write to temp file (Gradio handles cleanup)
            export_path = UPLOAD_PATH / "exports"
            export_path.mkdir(exist_ok=True)

            file_path = export_path / filename
            file_path.write_text(content, encoding='utf-8')

            return str(file_path), f"✅ Exported all {len(cards_data)} saved flashcards as {filename}"

    except Exception as e:
        print(f"Export error: {e}")
        return None, f"❌ Export error: {e}"

def clear_chat():
    """
    Clear chat history (in-session only).

    Returns empty list to reset Gradio chatbot.
    Phase 3.5 will delete from database.
    """
    return []

def build_ui():
    """Build the Gradio Blocks interface."""

    with gr.Blocks(
        title="StudyAgent - Art History"
    ) as demo:

        gr.Markdown(
            """
            # StudyAgent
            **Art History** · AI-powered study assistant. Upload course materials, chat with them, and generate flashcards.
            ---
            """
        )

        welcome_banner = gr.Markdown(
            "No classes yet. Click **+ New Class** above to get started.",
            visible=True
        )

        # Class selector group with inline new-class form
        with gr.Group():
            with gr.Row():
                class_selector = gr.Dropdown(
                    label="",
                    choices=[],
                    value=None,
                    allow_custom_value=False,
                    interactive=True,
                    scale=3
                )
                new_class_btn = gr.Button(
                    "+ New Class",
                    variant="primary",
                    size="sm",
                    scale=1
                )
                delete_class_btn = gr.Button(
                    "🗑️ Delete Class",
                    variant="stop",
                    size="sm",
                    scale=1
                )
            with gr.Row(visible=False) as new_class_container:
                new_class_input = gr.Textbox(
                    label="New Class Name",
                    placeholder="e.g., Art History 101",
                    interactive=True,
                    scale=2
                )
                create_class_btn = gr.Button(
                    "Create Class",
                    variant="primary",
                    size="sm",
                    scale=1
                )

        # Delete class confirmation modal
        with gr.Group(visible=False) as delete_class_modal:
            gr.Markdown("### ⚠️ Delete Class?")
            delete_class_warning = gr.Markdown()
            with gr.Row():
                confirm_delete_class_btn = gr.Button("Yes, Delete Everything", variant="stop")
                cancel_delete_class_btn = gr.Button("Cancel")

        # Hidden state
        current_class_name = gr.State(value=None)
        saved_msg = gr.State(value="")

        with gr.Tabs() as tabs:

            # === TAB 1: Classes & Materials ===
            with gr.Tab("📚 Classes & Materials", id="classes_tab"):
                file_upload = gr.File(
                    label="Upload Course Material",
                    file_types=['.pdf', '.txt', '.md', '.docx', '.pptx'],
                    height=120
                )
                with gr.Row():
                    input_name = gr.Textbox(
                        label="Document Name",
                        placeholder="e.g., Lecture 1: Renaissance",
                        scale=2
                    )
                    input_type = gr.Dropdown(
                        label="Material Type",
                        choices=['slides', 'textbook', 'notes', 'recording', 'quiz', 'study guide'],
                        value='slides',
                        scale=1
                    )
                upload_btn = gr.Button("Upload & Ingest", variant="primary")
                upload_status = gr.Markdown(value="")

                file_list = gr.Dataframe(
                    headers=["Name", "Type", "Status"],
                    label="Uploaded Materials",
                    interactive=False
                )

                with gr.Accordion(label="🔧 Advanced File Tools", open=False):

                    gr.Markdown("#### 📂 Manage Uploaded Files")

                    file_selector = gr.Dropdown(
                        label="Select file to delete",
                        choices=[],
                        interactive=True
                    )

                    with gr.Row():
                        delete_btn = gr.Button("Delete Selected", variant="stop", size="sm")
                        clear_all_btn = gr.Button("Clear All Files", variant="stop", size="sm")

                    with gr.Group(visible=False) as delete_confirm_modal:
                        gr.Markdown("### ⚠️ Confirm Deletion")
                        delete_confirm_text = gr.Markdown()
                        with gr.Row():
                            confirm_delete_btn = gr.Button("Yes, Delete", variant="stop")
                            cancel_delete_btn = gr.Button("Cancel")

                    with gr.Group(visible=False) as clear_confirm_modal:
                        gr.Markdown("### ⚠️ Delete ALL Files?")
                        gr.Markdown("This will delete all uploaded materials and cannot be undone!")
                        with gr.Row():
                            confirm_clear_btn = gr.Button("Yes, Clear All", variant="stop")
                            cancel_clear_btn = gr.Button("Cancel")

                    gr.Markdown("---")
                    gr.Markdown("#### 🔍 Improve Text Extraction (OCR)")
                    gr.Markdown("If your PDF contains scanned pages or images of text (not selectable text), use OCR to re-extract the content:")

                    ocr_file_selector = gr.Dropdown(
                        label="Select file to re-extract",
                        choices=[],
                        interactive=True
                    )

                    ocr_method_selector = gr.Radio(
                        label="OCR Method",
                        choices=["tesseract"],
                        value="tesseract",
                        info="Tesseract is free. Premium options require API keys."
                    )

                    retry_ocr_btn = gr.Button("🔄 Retry with OCR", variant="secondary", size="sm")
                    ocr_status = gr.Markdown(value="")

            # === TAB 2: Chat ===
            with gr.Tab("💬 Chat", id="chat_tab"):
                chatbot = gr.Chatbot(
                    label="AI Study Assistant",
                    height=500,
                    autoscroll=True,
                    editable=None,
                    buttons=["copy"],
                    placeholder="Ask a question about your uploaded materials, or upload some first from the Classes & Materials tab."
                )
                chat_input = gr.Textbox(
                    show_label=False,
                    placeholder="Ask a question... (Press Enter to send)",
                    lines=1,
                    max_lines=1,
                    submit_btn=True
                )
                with gr.Row():
                    chat_btn = gr.Button("Send", variant="primary")
                    clear_chat_btn = gr.Button("Clear Chat", variant="secondary", size="sm")

                with gr.Group(visible=False) as clear_chat_modal:
                    gr.Markdown("### ⚠️ Clear Chat?")
                    gr.Markdown("Clear all chat history for this class? This can't be undone.")
                    with gr.Row():
                        confirm_clear_chat_btn = gr.Button("Yes, Clear", variant="stop")
                        cancel_clear_chat_btn = gr.Button("Cancel")

            # === TAB 3: Flashcards & Export ===
            with gr.Tab("🃏 Flashcards & Export", id="flashcards_tab"):
                flashcard_topic = gr.Textbox(
                    label="Focus area (optional)",
                    placeholder="e.g., Baroque period — narrows the search, but results depend on your materials"
                )
                gen_flashcard_btn = gr.Button("✨ Generate Flashcards", variant="primary")

                flashcard_status = gr.Markdown(value="")

                flashcard_table = gr.Dataframe(
                    headers=["Term", "Definition"],
                    label="Generated Flashcards",
                    interactive=False,
                    wrap=True,
                    column_widths=["30%", "70%"]
                )

                export_format = gr.Radio(
                    label="Export Format",
                    choices=["Quizlet (TSV)", "Anki (CSV)"],
                    value="Quizlet (TSV)"
                )
                export_btn = gr.Button("💾 Export", variant="secondary")
                export_file = gr.File(label="Download", visible=False)

        # Helper function for updating file controls
        def update_file_controls(class_name: str):
            """Update file_list, file_selector, and ocr_file_selector after changes"""
            file_choices = get_file_choices(class_name)
            return (
                get_current_file_list(class_name),
                gr.update(choices=file_choices, value=None),
                gr.update(choices=file_choices, value=None)
            )

        # Event handlers

        # Chat helpers for optimistic display
        def append_user_message(message, history):
            """Immediately show user message in chat."""
            if not message or not message.strip():
                return history, "", message
            return history + [{"role": "user", "content": message}], "", message

        async def fetch_response(history, class_name, original_msg):
            """Call agent and append response."""
            if not original_msg or not original_msg.strip():
                return history
            prior = history[:-1]  # Remove user msg we just appended
            return await chat_with_materials(original_msg, prior, class_name)

        # "+ New Class" button — show inline form
        new_class_btn.click(
            fn=lambda: (gr.update(visible=True), gr.update(value="➕ Create New Class...")),
            outputs=[new_class_container, class_selector]
        )

        # Class selection event — reset statuses across tabs
        class_selector.change(
            fn=handle_class_selection,
            inputs=[class_selector],
            outputs=[
                current_class_name,
                new_class_container,
                create_class_btn,
                file_list,
                file_selector,
                ocr_file_selector
            ]
        ).then(
            fn=lambda cls: gr.update(visible=(cls is None or cls == "➕ Create New Class...")),
            inputs=[class_selector],
            outputs=[welcome_banner]
        ).then(
            fn=lambda: ("", ""),
            outputs=[upload_status, flashcard_status]
        )

        # Create new class event — auto-navigate to Classes tab
        create_class_btn.click(
            fn=create_new_class,
            inputs=[new_class_input],
            outputs=[
                class_selector,
                new_class_input,
                create_class_btn,
                upload_status
            ]
        ).then(
            fn=handle_class_selection,
            inputs=[class_selector],
            outputs=[
                current_class_name,
                new_class_container,
                create_class_btn,
                file_list,
                file_selector,
                ocr_file_selector
            ]
        ).then(
            fn=lambda: (gr.update(visible=False), "", gr.Tabs(selected="classes_tab")),
            outputs=[welcome_banner, flashcard_status, tabs]
        )

        # Delete class — two-step confirmation
        def show_delete_class_warning(class_name):
            if not class_name or class_name == "➕ Create New Class...":
                return gr.update(visible=False), ""
            return (
                gr.update(visible=True),
                f"This will permanently delete **{class_name}** and all its files, flashcards, and chat history."
            )

        delete_class_btn.click(
            fn=show_delete_class_warning,
            inputs=[current_class_name],
            outputs=[delete_class_modal, delete_class_warning]
        )

        cancel_delete_class_btn.click(
            fn=lambda: gr.update(visible=False),
            outputs=[delete_class_modal]
        )

        confirm_delete_class_btn.click(
            fn=delete_class,
            inputs=[current_class_name],
            outputs=[
                upload_status,
                class_selector,
                file_list,
                new_class_input,
                create_class_btn,
                current_class_name
            ],
            show_progress=True
        ).then(
            fn=lambda: (gr.update(visible=False), gr.update(visible=True), ""),
            outputs=[delete_class_modal, welcome_banner, flashcard_status]
        )

        # Auto-populate document name from filename
        def auto_populate_name(file):
            if file is None:
                return ""
            name = Path(file.name).stem
            name = name.replace('_', ' ').replace('-', ' ')
            return name.title()

        file_upload.change(
            fn=auto_populate_name,
            inputs=[file_upload],
            outputs=[input_name]
        )

        # Upload file event
        upload_btn.click(
            fn=upload_file,
            inputs=[file_upload, current_class_name, input_name, input_type],
            outputs=[upload_status, file_list],
            show_progress="hidden"
        ).then(
            fn=lambda: gr.update(choices=get_all_classes()),
            outputs=[class_selector]
        ).then(
            fn=update_file_controls,
            inputs=[current_class_name],
            outputs=[file_list, file_selector, ocr_file_selector]
        )

        # Send message handler (optimistic display)
        chat_btn.click(
            fn=append_user_message,
            inputs=[chat_input, chatbot],
            outputs=[chatbot, chat_input, saved_msg]
        ).then(
            fn=fetch_response,
            inputs=[chatbot, current_class_name, saved_msg],
            outputs=[chatbot]
        )

        # Clear chat — two-step confirmation
        clear_chat_btn.click(
            fn=lambda: gr.update(visible=True),
            outputs=[clear_chat_modal]
        )

        cancel_clear_chat_btn.click(
            fn=lambda: gr.update(visible=False),
            outputs=[clear_chat_modal]
        )

        confirm_clear_chat_btn.click(
            fn=clear_chat,
            outputs=[chatbot]
        ).then(
            fn=lambda: gr.update(visible=False),
            outputs=[clear_chat_modal]
        )

        # Submit on Enter key (optimistic display)
        chat_input.submit(
            fn=append_user_message,
            inputs=[chat_input, chatbot],
            outputs=[chatbot, chat_input, saved_msg]
        ).then(
            fn=fetch_response,
            inputs=[chatbot, current_class_name, saved_msg],
            outputs=[chatbot]
        )

        # Individual file delete workflow
        def show_delete_confirmation(selected_input_id, class_name):
            """Show confirmation modal with file name."""
            if not selected_input_id:
                return gr.update(visible=False), "", "⚠️ Please select a file to delete."
            choices_dict = {v: k for k, v in get_file_choices(class_name)}
            file_name = choices_dict.get(selected_input_id, 'Unknown')
            return gr.update(visible=True), f"Delete **{file_name}**?", ""

        delete_btn.click(
            fn=show_delete_confirmation,
            inputs=[file_selector, current_class_name],
            outputs=[delete_confirm_modal, delete_confirm_text, upload_status]
        )

        cancel_delete_btn.click(
            fn=lambda: gr.update(visible=False),
            outputs=[delete_confirm_modal]
        )

        confirm_delete_btn.click(
            fn=delete_file,
            inputs=[file_selector, current_class_name],
            outputs=[upload_status, file_list]
        ).then(
            fn=lambda: gr.update(visible=False),
            outputs=[delete_confirm_modal]
        ).then(
            fn=update_file_controls,
            inputs=[current_class_name],
            outputs=[file_list, file_selector, ocr_file_selector]
        )

        # Clear all files workflow
        clear_all_btn.click(
            fn=lambda: gr.update(visible=True),
            outputs=[clear_confirm_modal]
        )

        cancel_clear_btn.click(
            fn=lambda: gr.update(visible=False),
            outputs=[clear_confirm_modal]
        )

        confirm_clear_btn.click(
            fn=clear_all_files,
            inputs=[current_class_name],
            outputs=[upload_status, file_list]
        ).then(
            fn=lambda: gr.update(visible=False),
            outputs=[clear_confirm_modal]
        ).then(
            fn=update_file_controls,
            inputs=[current_class_name],
            outputs=[file_list, file_selector, ocr_file_selector]
        )

        # OCR retry handler
        retry_ocr_btn.click(
            fn=retry_with_ocr,
            inputs=[ocr_file_selector, ocr_method_selector, current_class_name],
            outputs=[ocr_status, file_list]
        )

        gen_flashcard_btn.click(
            fn=generate_flashcards,
            inputs=[current_class_name, flashcard_topic],
            outputs=[flashcard_status, flashcard_table]
        )

        # Export with feedback and show/hide download
        export_btn.click(
            fn=export_flashcards,
            inputs=[current_class_name, export_format],
            outputs=[export_file, flashcard_status]
        ).then(
            fn=lambda f: gr.update(visible=(f is not None)),
            inputs=[export_file],
            outputs=[export_file]
        )

        # Helper function for OCR methods
        def get_ocr_methods():
            from app.pipelines.ocr import get_available_ocr_methods
            methods = get_available_ocr_methods()
            if not methods:
                methods = ['tesseract (install required)']
            return gr.update(choices=methods, value=methods[0] if methods else None)

        # Initialize dropdown on app load
        demo.load(
            fn=lambda: gr.update(choices=get_all_classes()),
            outputs=[class_selector]
        )

        # Initialize OCR method selector on app load
        demo.load(
            fn=get_ocr_methods,
            outputs=[ocr_method_selector]
        )

    return demo

if __name__ == '__main__':
    init_app()
    demo = build_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
        css="""
            table td, table th,
            .cell-wrap, .header-wrap,
            .gradio-dataframe td, .gradio-dataframe th {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
            }
        """
    )
