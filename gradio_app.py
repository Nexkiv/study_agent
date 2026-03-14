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

from app.config import validate_config, ensure_directories
from app.extensions import db, get_chroma_client
from app.models import Class, Input, Flashcard, ChatMessage
from app.utils.file_handler import save_upload
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
    """Handle file upload (Phase 1: save only, no ingestion yet)."""
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

            # Create input record
            input_obj = Input(
                class_id=class_obj.id,
                name=input_name,
                input_type=input_type,
                file_path=relative_path,
                raw_text=None  # Phase 2 will populate this
            )
            db.session.add(input_obj)
            db.session.commit()

            # Get updated file list
            inputs = Input.query.filter_by(class_id=class_obj.id).all()
            file_list = [[inp.name, inp.input_type, "Saved (not ingested yet)"]
                        for inp in inputs]

            return f"✅ Uploaded: {input_name}", file_list

    except Exception as e:
        return f"❌ Error: {str(e)}", None

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

        class_name_input = gr.Textbox(
            label="Class Name",
            placeholder="e.g., Art History 101",
            value="Art History 101"
        )

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

        # Event handlers
        upload_btn.click(
            fn=upload_file,
            inputs=[file_upload, class_name_input, input_name, input_type],
            outputs=[upload_status, file_list]
        )

        chat_btn.click(
            fn=chat_with_materials,
            inputs=[chat_input, chatbot, class_name_input],
            outputs=[chatbot]
        )

        gen_flashcard_btn.click(
            fn=generate_flashcards,
            inputs=[class_name_input, flashcard_topic],
            outputs=[flashcard_status, flashcard_table]
        )

        export_btn.click(
            fn=export_flashcards,
            inputs=[class_name_input, export_format],
            outputs=[export_file]
        )

        gr.Markdown(
            """
            ---
            **Phase 1**: Upload files, save to database ✅
            **Phase 2**: Text extraction, chunking, embedding 🚧
            **Phase 3**: RAG chat agent with tool use 🚧
            **Phase 4**: Flashcard generation with structured outputs 🚧
            """
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
