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
from app.pipelines.ingestion import process_upload
from app.pipelines.chunking import chunk_text, generate_embeddings, delete_embeddings
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
                raw_text = process_upload(
                    str(absolute_path),
                    class_obj.id,
                    input_name,
                    input_type
                )
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
                raw_text=raw_text
            )
            db.session.add(input_obj)
            db.session.commit()

            # Get updated file list
            inputs = Input.query.filter_by(class_id=class_obj.id).all()
            file_list = [[inp.name, inp.input_type, "✅ Ingested"]
                        for inp in inputs]

            chunk_count = len(chunks)
            return (
                f"✅ Uploaded: {input_name}\n"
                f"📄 Extracted {len(raw_text):,} characters\n"
                f"🔢 Created {chunk_count} chunks\n"
                f"💾 Stored embeddings in ChromaDB"
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

    Returns: (class_name, new_class_visible, create_btn_visible, file_list, file_selector)
    """
    if selected_value == "➕ Create New Class...":
        return (
            None,  # class_name (hidden state)
            gr.update(visible=True),  # new_class_input
            gr.update(visible=True),  # create_class_btn
            [],  # file_list (empty)
            gr.update(choices=[], value=None)  # file_selector (empty)
        )
    else:
        # Existing class selected
        return (
            selected_value,  # class_name
            gr.update(visible=False),  # new_class_input
            gr.update(visible=False),  # create_class_btn
            get_current_file_list(selected_value),  # file_list
            gr.update(choices=get_file_choices(selected_value), value=None)  # file_selector
        )

def create_new_class(new_class_name: str):
    """
    Create a new class and update dropdown choices.

    Returns: (updated_choices, selected_value, new_class_visible, create_btn_visible, status)
    """
    if not new_class_name or not new_class_name.strip():
        return (
            get_all_classes(),  # Keep current choices
            "➕ Create New Class...",  # Keep on create option
            gr.update(visible=True),  # Keep input visible
            gr.update(visible=True),  # Keep button visible
            "⚠️ Please enter a class name"  # status
        )

    with app.app_context():
        # Check if class already exists
        existing = Class.query.filter_by(name=new_class_name).first()
        if existing:
            return (
                get_all_classes(),
                new_class_name,  # Select the existing class
                gr.update(visible=False),
                gr.update(visible=False),
                f"✅ Switched to existing class: {new_class_name}"
            )

        # Create new class
        new_class = Class(name=new_class_name)
        db.session.add(new_class)
        db.session.commit()

        # Update dropdown choices
        updated_choices = get_all_classes()

        return (
            updated_choices,  # Dropdown choices with new class
            new_class_name,  # Select the newly created class
            gr.update(visible=False),  # Hide new class input
            gr.update(visible=False),  # Hide create button
            f"✅ Created new class: {new_class_name}"  # status
        )

def chat_with_materials(message, history, class_name):
    """RAG chat agent (Phase 3 placeholder)."""
    if not class_name:
        # Gradio 5.x format: list of dicts with 'role' and 'content'
        return history + [{"role": "user", "content": message},
                         {"role": "assistant", "content": "⚠️ Please enter a class name first"}]

    response = (
        f"🚧 Phase 3 feature coming soon!\n\n"
        f"You asked: '{message}'\n"
        f"Class: {class_name}\n\n"
        f"The RAG agent will search your uploaded materials and answer."
    )

    # Gradio 5.x format
    return history + [{"role": "user", "content": message},
                     {"role": "assistant", "content": response}]

def generate_flashcards(class_name, topic):
    """Generate flashcards (Phase 4 placeholder)."""
    if not class_name:
        return "⚠️ Please enter a class name", None

    return (
        f"🚧 Phase 4 feature coming soon!\n\n"
        f"Will generate flashcards for {class_name}"
        f"{f' on topic: {topic}' if topic else ''}\n"
        f"Using OpenAI Structured Outputs."
    ), None

def export_flashcards(class_name, format_choice):
    """Export flashcards (Phase 4 placeholder)."""
    return None

def build_ui():
    """Build the Gradio Blocks interface."""

    with gr.Blocks(
        title="StudyAgent - Art History",
        theme=gr.themes.Soft(primary_hue="indigo")
    ) as demo:

        gr.Markdown(
            """
            # 🎨 StudyAgent - Art History Study Tool

            **Hybrid AI Architecture**: Deterministic pipelines + agentic features

            Upload PDFs → Chat with materials via RAG → Generate flashcards → Export to Quizlet
            """
        )

        class_selector = gr.Dropdown(
            label="Select Class",
            choices=[],  # Populated by get_all_classes() on load
            value=None,  # Start with nothing selected
            allow_custom_value=False,
            interactive=True
        )

        new_class_input = gr.Textbox(
            label="New Class Name",
            placeholder="e.g., Art History 101",
            visible=False,
            interactive=True
        )

        create_class_btn = gr.Button(
            "Create Class",
            variant="primary",
            size="sm",
            visible=False
        )

        # Hidden state to track current class name
        current_class_name = gr.State(value=None)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Upload Materials")

                file_upload = gr.File(
                    label="Select PDF or Text File",
                    file_types=['.pdf', '.txt', '.md', '.docx']
                )
                input_name = gr.Textbox(
                    label="Name this input",
                    placeholder="e.g., Lecture 1: Renaissance"
                )
                input_type = gr.Dropdown(
                    label="Input Type",
                    choices=['slides', 'textbook', 'notes', 'recording', 'quiz'],
                    value='slides'
                )
                upload_btn = gr.Button("Upload & Ingest", variant="primary")
                upload_status = gr.Textbox(label="Status", interactive=False)

                file_list = gr.Dataframe(
                    headers=["Name", "Type", "Status"],
                    label="Uploaded Materials",
                    interactive=False
                )

                # File deletion controls
                gr.Markdown("---")
                gr.Markdown("#### 🗑️ File Management")

                file_selector = gr.Dropdown(
                    label="Select file to delete",
                    choices=[],
                    interactive=True
                )

                with gr.Row():
                    delete_btn = gr.Button("Delete Selected", variant="stop", size="sm")
                    clear_all_btn = gr.Button("Clear All Files", variant="stop", size="sm")

                # Confirmation modals (hidden by default)
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

            with gr.Column(scale=2):
                gr.Markdown("### 💬 Chat with Your Materials")

                chatbot = gr.Chatbot(
                    label="RAG Agent",
                    height=400
                )
                chat_input = gr.Textbox(
                    label="Ask a question",
                    placeholder="e.g., What are key characteristics of Baroque art?"
                )
                chat_btn = gr.Button("Send", variant="primary")

            with gr.Column(scale=1):
                gr.Markdown("### 🃏 Flashcards")

                flashcard_topic = gr.Textbox(
                    label="Topic (optional)",
                    placeholder="e.g., Baroque period"
                )
                gen_flashcard_btn = gr.Button("✨ Generate Flashcards", variant="primary")

                flashcard_status = gr.Textbox(label="Status", interactive=False)

                flashcard_table = gr.Dataframe(
                    headers=["Term", "Definition"],
                    label="Generated Flashcards",
                    interactive=False
                )

                export_format = gr.Radio(
                    label="Export Format",
                    choices=["Quizlet (TSV)", "Anki (CSV)"],
                    value="Quizlet (TSV)"
                )
                export_btn = gr.Button("💾 Export")
                export_file = gr.File(label="Download")

        # Helper function for updating file controls
        def update_file_controls(class_name: str):
            """Update both file_list and file_selector after changes"""
            return (
                get_current_file_list(class_name),
                gr.update(choices=get_file_choices(class_name), value=None)
            )

        # Event handlers

        # Class selection event
        class_selector.change(
            fn=handle_class_selection,
            inputs=[class_selector],
            outputs=[
                current_class_name,  # Hidden state
                new_class_input,     # Show/hide new class input
                create_class_btn,    # Show/hide create button
                file_list,           # Update file list
                file_selector        # Update file selector
            ]
        )

        # Create new class event
        create_class_btn.click(
            fn=create_new_class,
            inputs=[new_class_input],
            outputs=[
                class_selector,      # Update choices
                class_selector,      # Update selected value (new class)
                new_class_input,     # Hide input
                create_class_btn,    # Hide button
                upload_status        # Show status message
            ]
        ).then(
            fn=handle_class_selection,  # Trigger selection handler for new class
            inputs=[class_selector],
            outputs=[
                current_class_name,
                new_class_input,
                create_class_btn,
                file_list,
                file_selector
            ]
        )

        # Upload file event
        upload_btn.click(
            fn=upload_file,
            inputs=[file_upload, current_class_name, input_name, input_type],
            outputs=[upload_status, file_list],
            show_progress="hidden"  # Disable automatic output component highlighting
        ).then(
            fn=lambda: gr.update(choices=get_all_classes()),  # Refresh dropdown
            outputs=[class_selector]
        ).then(
            fn=update_file_controls,
            inputs=[current_class_name],
            outputs=[file_list, file_selector]
        )

        chat_btn.click(
            fn=chat_with_materials,
            inputs=[chat_input, chatbot, current_class_name],
            outputs=[chatbot]
        )

        # Individual delete workflow
        def show_delete_confirmation(selected_input_id, class_name):
            """Show confirmation modal with file name"""
            if not selected_input_id:
                return gr.update(visible=False), ""

            # Get file name from choices
            choices_dict = dict(get_file_choices(class_name))
            file_name = choices_dict.get(selected_input_id, 'Unknown')
            return gr.update(visible=True), f"Delete **{file_name}**?"

        delete_btn.click(
            fn=show_delete_confirmation,
            inputs=[file_selector, current_class_name],
            outputs=[delete_confirm_modal, delete_confirm_text]
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
            outputs=[file_list, file_selector]
        )

        # Clear all workflow
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
            outputs=[file_list, file_selector]
        )

        gen_flashcard_btn.click(
            fn=generate_flashcards,
            inputs=[current_class_name, flashcard_topic],
            outputs=[flashcard_status, flashcard_table]
        )

        export_btn.click(
            fn=export_flashcards,
            inputs=[current_class_name, export_format],
            outputs=[export_file]
        )

        gr.Markdown(
            """
            ---
            **Phase 1**: Upload files, save to database ✅
            **Phase 2**: Text extraction, chunking, embedding ✅
            **Phase 3**: RAG chat agent with tool use 🚧
            **Phase 4**: Flashcard generation with structured outputs 🚧
            """
        )

        # Initialize dropdown on app load
        demo.load(
            fn=lambda: gr.update(choices=get_all_classes()),
            outputs=[class_selector]
        )

    return demo

if __name__ == '__main__':
    init_app()
    demo = build_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False
    )
