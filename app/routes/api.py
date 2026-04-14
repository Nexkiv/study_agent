"""
REST API routes for StudyAgent.

Returns JSON for HTMX/JS consumption.
"""
import asyncio
import io
import shutil
from pathlib import Path

from flask import Blueprint, request, jsonify, send_file
from app.extensions import db, get_chroma_client
from app.models import Class, Input, FlashcardSet, Flashcard, ChatMessage
from app.config import UPLOAD_PATH
from app.utils.file_handler import save_upload, delete_upload
from app.pipelines.ingestion import process_upload, extract_pdf
from app.pipelines.chunking import chunk_text, generate_embeddings, delete_embeddings
from app.pipelines.exporters import export_to_quizlet, export_to_anki
from app.agents.chat_agent import (
    get_async_openai_client,
    create_rag_agent_config,
    create_search_tool,
    create_list_sections_tool,
    execute_python,
    correct_spelling,
    search_web,
)
from app.agents.run_agent import run_agent, cancel_agent, reset_cancel, AgentCancelled
from app.agents.tools import ToolBox
from app.agents.study_agent import generate_flashcards_for_topic

api_bp = Blueprint('api', __name__)


# ---------------------------------------------------------------------------
# Class management
# ---------------------------------------------------------------------------

@api_bp.route('/classes', methods=['GET'])
def list_classes():
    """Get all classes with file counts."""
    classes = Class.query.order_by(Class.name).all()
    result = []
    for cls in classes:
        file_count = Input.query.filter_by(class_id=cls.id).count()
        result.append({
            'id': cls.id,
            'name': cls.name,
            'file_count': file_count,
        })
    return jsonify(result)


@api_bp.route('/classes', methods=['POST'])
def create_class():
    """Create a new class."""
    data = request.get_json()
    name = (data.get('name') or '').strip()

    if not name:
        return jsonify({'error': 'Please enter a class name.'}), 400

    existing = Class.query.filter_by(name=name).first()
    if existing:
        return jsonify({
            'id': existing.id,
            'name': existing.name,
            'message': f'Switched to existing class: {name}.',
            'existed': True,
        })

    new_class = Class(name=name)
    db.session.add(new_class)
    db.session.commit()

    return jsonify({
        'id': new_class.id,
        'name': new_class.name,
        'message': f'Created new class: {name}.',
        'existed': False,
    }), 201


@api_bp.route('/classes/<int:class_id>', methods=['DELETE'])
def delete_class(class_id):
    """Delete a class and all related data (cascade)."""
    class_obj = Class.query.get(class_id)
    if not class_obj:
        return jsonify({'error': 'Class not found.'}), 404

    class_name = class_obj.name

    # Delete physical files
    class_upload_dir = UPLOAD_PATH / str(class_obj.id)
    if class_upload_dir.exists():
        shutil.rmtree(class_upload_dir)

    # Delete ChromaDB collection
    try:
        client = get_chroma_client()
        client.delete_collection(f"class_{class_obj.id}")
    except Exception:
        pass

    # Delete database records (cascade handles related records)
    db.session.delete(class_obj)
    db.session.commit()

    return jsonify({'message': f"Class '{class_name}' deleted successfully."})


@api_bp.route('/classes/<int:class_id>', methods=['PATCH'])
def rename_class(class_id):
    """Rename a class."""
    class_obj = Class.query.get(class_id)
    if not class_obj:
        return jsonify({'error': 'Class not found.'}), 404

    data = request.get_json()
    new_name = (data.get('name') or '').strip()
    if not new_name:
        return jsonify({'error': 'Please enter a class name.'}), 400

    class_obj.name = new_name
    db.session.commit()
    return jsonify({'id': class_obj.id, 'name': class_obj.name})


# ---------------------------------------------------------------------------
# File management
# ---------------------------------------------------------------------------

@api_bp.route('/classes/<int:class_id>/files', methods=['POST'])
def upload_file(class_id):
    """Upload and ingest a file into a class."""
    class_obj = Class.query.get(class_id)
    if not class_obj:
        return jsonify({'error': 'Class not found.'}), 404

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided.'}), 400

    file = request.files['file']
    if not file.filename:
        return jsonify({'error': 'No file selected.'}), 400

    input_name = (request.form.get('name') or '').strip()
    input_type = request.form.get('type', 'slides')

    if not input_name:
        return jsonify({'error': 'Please enter a document name.'}), 400

    try:
        # Extract original filename
        original_filename = file.filename

        # Save file to disk
        relative_path = save_upload(file, class_obj.id, original_filename)
        absolute_path = UPLOAD_PATH / relative_path

        # Extract text
        ext = Path(original_filename).suffix.lower()
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

        # Chunk and embed
        chunks = chunk_text(raw_text)
        generate_embeddings(class_obj.id, input_name, chunks)

        # Create input record
        input_obj = Input(
            class_id=class_obj.id,
            name=input_name,
            input_type=input_type,
            file_path=relative_path,
            raw_text=raw_text,
            extraction_method=extraction_method,
        )
        db.session.add(input_obj)
        db.session.commit()

        result = {
            'id': input_obj.id,
            'name': input_name,
            'type': input_type,
            'chars': len(raw_text),
            'chunks': len(chunks),
            'needs_ocr': needs_ocr,
        }
        return jsonify(result), 201

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api_bp.route('/classes/<int:class_id>/inputs', methods=['GET'])
def list_inputs(class_id):
    """Get all inputs (uploaded files) for a class."""
    inputs = Input.query.filter_by(class_id=class_id).order_by(Input.created_at.desc()).all()
    return jsonify([{
        'id': inp.id,
        'name': inp.name,
        'type': inp.input_type,
        'extraction_method': inp.extraction_method,
    } for inp in inputs])


@api_bp.route('/ocr-methods', methods=['GET'])
def get_ocr_methods():
    """Get available OCR methods."""
    try:
        from app.pipelines.ocr import get_available_ocr_methods
        methods = get_available_ocr_methods()
    except Exception:
        methods = ['tesseract']
    if not methods:
        methods = ['tesseract']
    return jsonify(methods)


@api_bp.route('/files/<int:file_id>', methods=['DELETE'])
def delete_file(file_id):
    """Delete a single uploaded file and related data."""
    input_obj = Input.query.get(file_id)
    if not input_obj:
        return jsonify({'error': 'File not found.'}), 404

    input_name = input_obj.name
    file_path = input_obj.file_path
    class_id = input_obj.class_id

    # Delete database records
    try:
        Flashcard.query.filter_by(input_id=file_id).delete()
        db.session.delete(input_obj)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Database error: {e}'}), 500

    # Delete ChromaDB chunks (best-effort)
    try:
        delete_embeddings(class_id, input_name)
    except Exception:
        pass

    # Delete disk file (best-effort)
    if file_path:
        try:
            delete_upload(file_path)
        except Exception:
            pass

    return jsonify({'message': f'Deleted: {input_name}'})


@api_bp.route('/classes/<int:class_id>/files', methods=['DELETE'])
def clear_all_files(class_id):
    """Delete ALL files for a class."""
    class_obj = Class.query.get(class_id)
    if not class_obj:
        return jsonify({'error': 'Class not found.'}), 404

    inputs = Input.query.filter_by(class_id=class_id).all()
    if not inputs:
        return jsonify({'message': 'No files to delete.'}), 200

    count = len(inputs)
    file_data = [(inp.id, inp.name, inp.file_path) for inp in inputs]

    # Delete database records
    try:
        for input_id, _, _ in file_data:
            Flashcard.query.filter_by(input_id=input_id).delete()
        Input.query.filter_by(class_id=class_id).delete()
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Database error: {e}'}), 500

    # Best-effort cleanup
    for _, input_name, _ in file_data:
        try:
            delete_embeddings(class_id, input_name)
        except Exception:
            pass
    for _, _, fp in file_data:
        if fp:
            try:
                delete_upload(fp)
            except Exception:
                pass

    return jsonify({'message': f'Cleared {count} file(s).'})


@api_bp.route('/files/<int:file_id>/ocr', methods=['POST'])
def retry_ocr(file_id):
    """Re-process a file using OCR."""
    input_obj = Input.query.get(file_id)
    if not input_obj:
        return jsonify({'error': 'File not found.'}), 404

    data = request.get_json() or {}
    ocr_method = data.get('method', 'tesseract')

    file_path = UPLOAD_PATH / input_obj.file_path

    from app.pipelines.ocr import (
        extract_pdf_with_tesseract,
        extract_pdf_with_mathpix,
        extract_pdf_with_claude_vision,
    )

    try:
        if ocr_method == 'tesseract':
            raw_text = extract_pdf_with_tesseract(str(file_path))
        elif ocr_method == 'mathpix':
            raw_text = extract_pdf_with_mathpix(str(file_path))
        elif ocr_method == 'claude':
            raw_text = extract_pdf_with_claude_vision(str(file_path))
        else:
            return jsonify({'error': f'Unknown OCR method: {ocr_method}'}), 400
    except RuntimeError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'OCR failed: {e}'}), 500

    # Re-chunk and re-embed
    delete_embeddings(input_obj.class_id, input_obj.name)
    chunks = chunk_text(raw_text)
    generate_embeddings(input_obj.class_id, input_obj.name, chunks)

    # Update database
    input_obj.raw_text = raw_text
    input_obj.extraction_method = ocr_method
    db.session.commit()

    return jsonify({
        'message': f'Re-extracted with {ocr_method.capitalize()} OCR',
        'chars': len(raw_text),
        'chunks': len(chunks),
    })


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

@api_bp.route('/classes/<int:class_id>/chat', methods=['GET'])
def get_chat_history(class_id):
    """Get chat history for a class."""
    messages = ChatMessage.query.filter_by(class_id=class_id)\
        .order_by(ChatMessage.created_at).all()
    return jsonify([m.to_dict() for m in messages])


@api_bp.route('/classes/<int:class_id>/chat', methods=['POST'])
def chat(class_id):
    """Send a chat message and get AI response."""
    class_obj = Class.query.get(class_id)
    if not class_obj:
        return jsonify({'error': 'Class not found.'}), 404

    data = request.get_json()
    message = (data.get('message') or '').strip()
    if not message:
        return jsonify({'error': 'Message cannot be empty.'}), 400

    class_name = class_obj.name

    try:
        # Correct spelling
        corrected_message = correct_spelling(message, class_name)

        # Initialize tools
        toolbox = ToolBox()
        search_tool = create_search_tool(class_name)
        list_sections_tool = create_list_sections_tool(class_name)
        toolbox.tool(search_tool)
        toolbox.tool(list_sections_tool)
        toolbox.tool(search_web)
        toolbox.tool(execute_python)

        client = get_async_openai_client()
        agent_config = create_rag_agent_config(class_name)

        # Build conversation history from database
        history_messages = ChatMessage.query.filter_by(class_id=class_id)\
            .order_by(ChatMessage.created_at).all()
        conversation_history = [m.to_dict() for m in history_messages]

        # Run agent
        response_text = asyncio.run(run_agent(
            client=client,
            toolbox=toolbox,
            agent=agent_config,
            user_message=corrected_message,
            history=conversation_history,
            usage=None,
        ))

        final_response = response_text or "I apologize, but I couldn't generate a response."

        # Save to database
        user_msg = ChatMessage(class_id=class_id, role='user', content=message)
        assistant_msg = ChatMessage(class_id=class_id, role='assistant', content=final_response)
        db.session.add(user_msg)
        db.session.add(assistant_msg)
        db.session.commit()

        return jsonify({
            'user': {'role': 'user', 'content': message},
            'assistant': {'role': 'assistant', 'content': final_response},
        })

    except Exception as e:
        error_msg = f"Error: {e}"
        # Still save the failed exchange
        user_msg = ChatMessage(class_id=class_id, role='user', content=message)
        assistant_msg = ChatMessage(class_id=class_id, role='assistant', content=error_msg)
        db.session.add(user_msg)
        db.session.add(assistant_msg)
        db.session.commit()

        return jsonify({
            'user': {'role': 'user', 'content': message},
            'assistant': {'role': 'assistant', 'content': error_msg},
        })


@api_bp.route('/classes/<int:class_id>/chat', methods=['DELETE'])
def clear_chat(class_id):
    """Clear all chat history for a class."""
    ChatMessage.query.filter_by(class_id=class_id).delete()
    db.session.commit()
    return jsonify({'message': 'Chat history cleared.'})


# ---------------------------------------------------------------------------
# Flashcard Sets
# ---------------------------------------------------------------------------

@api_bp.route('/classes/<int:class_id>/flashcard-sets', methods=['GET'])
def list_flashcard_sets(class_id):
    """Get all flashcard sets for a class."""
    sets = FlashcardSet.query.filter_by(class_id=class_id)\
        .order_by(FlashcardSet.created_at.desc()).all()
    return jsonify([s.to_dict() for s in sets])


@api_bp.route('/flashcard-sets/<int:set_id>', methods=['DELETE'])
def delete_flashcard_set(set_id):
    """Delete a flashcard set and all its cards."""
    fs = FlashcardSet.query.get(set_id)
    if not fs:
        return jsonify({'error': 'Set not found.'}), 404
    name = fs.name
    db.session.delete(fs)  # cascade deletes cards
    db.session.commit()
    return jsonify({'message': f'Deleted set: {name}'})


@api_bp.route('/flashcard-sets/<int:set_id>', methods=['PATCH'])
def rename_flashcard_set(set_id):
    """Rename a flashcard set."""
    fs = FlashcardSet.query.get(set_id)
    if not fs:
        return jsonify({'error': 'Set not found.'}), 404
    data = request.get_json()
    new_name = (data.get('name') or '').strip()
    if not new_name:
        return jsonify({'error': 'Name cannot be empty.'}), 400
    fs.name = new_name
    db.session.commit()
    return jsonify(fs.to_dict())


# ---------------------------------------------------------------------------
# Flashcards
# ---------------------------------------------------------------------------

@api_bp.route('/classes/<int:class_id>/flashcards', methods=['GET'])
def get_flashcards(class_id):
    """Get flashcards for a class, optionally filtered by set."""
    set_id = request.args.get('set_id', type=int)
    query = Flashcard.query.filter_by(class_id=class_id)
    if set_id:
        query = query.filter_by(set_id=set_id)
    flashcards = query.order_by(Flashcard.created_at.desc()).all()
    return jsonify([f.to_dict() for f in flashcards])


@api_bp.route('/classes/<int:class_id>/flashcards', methods=['POST'])
def generate_flashcards(class_id):
    """Generate flashcards and auto-create a set."""
    class_obj = Class.query.get(class_id)
    if not class_obj:
        return jsonify({'error': 'Class not found.'}), 404

    if not Input.query.filter_by(class_id=class_id).first():
        return jsonify({'error': 'No materials uploaded yet. Please upload files first.'}), 400

    data = request.get_json() or {}
    topic = (data.get('topic') or '').strip()

    # Determine set name from topic
    if topic:
        set_name = topic
        agent_topic = topic
    else:
        from datetime import datetime
        set_name = f"All topics - {datetime.now().strftime('%b %d')}"
        agent_topic = "all major topics covered in the course materials"

    try:
        reset_cancel()
        flashcards, _ = asyncio.run(
            generate_flashcards_for_topic(class_obj.name, agent_topic)
        )

        if not flashcards:
            return jsonify({'error': 'No flashcards generated. Try a more specific topic.'}), 400

        # Create set
        new_set = FlashcardSet(class_id=class_id, name=set_name)
        db.session.add(new_set)
        db.session.flush()  # get the set ID

        # Save flashcards linked to set
        for card in flashcards:
            fc = Flashcard(
                class_id=class_id,
                set_id=new_set.id,
                input_id=None,
                term=card['term'],
                definition=card['definition'],
            )
            db.session.add(fc)
        db.session.commit()

        total_count = Flashcard.query.filter_by(class_id=class_id).count()

        return jsonify({
            'generated': len(flashcards),
            'total': total_count,
            'flashcards': flashcards,
            'class_name': class_obj.name,
            'set_id': new_set.id,
            'set_name': new_set.name,
        }), 201

    except AgentCancelled:
        return jsonify({'cancelled': True, 'message': 'Generation cancelled.'}), 200

    except Exception as e:
        error_str = str(e)
        if "context_length_exceeded" in error_str or "context window" in error_str:
            return jsonify({
                'error': 'Your materials are too large to process at once. '
                         'Try a more specific topic, or remove some files and try again.'
            }), 400
        return jsonify({'error': 'Something went wrong generating flashcards. Please try again.'}), 500


@api_bp.route('/flashcards/cancel', methods=['POST'])
def cancel_flashcard_generation():
    """Cancel an in-progress flashcard generation."""
    cancel_agent()
    return jsonify({'message': 'Cancel signal sent.'}), 200


@api_bp.route('/flashcards/<int:flashcard_id>', methods=['PUT'])
def update_flashcard(flashcard_id):
    """Update a flashcard's term or definition."""
    fc = Flashcard.query.get(flashcard_id)
    if not fc:
        return jsonify({'error': 'Flashcard not found.'}), 404
    data = request.get_json()
    if 'term' in data:
        fc.term = data['term']
    if 'definition' in data:
        fc.definition = data['definition']
    db.session.commit()
    return jsonify(fc.to_dict())


@api_bp.route('/flashcards/<int:flashcard_id>', methods=['DELETE'])
def delete_flashcard(flashcard_id):
    """Delete a single flashcard."""
    fc = Flashcard.query.get(flashcard_id)
    if not fc:
        return jsonify({'error': 'Flashcard not found.'}), 404
    db.session.delete(fc)
    db.session.commit()
    return jsonify({'message': 'Flashcard deleted.'})


@api_bp.route('/classes/<int:class_id>/export', methods=['GET'])
def export_flashcards(class_id):
    """Export flashcards as a downloadable file, optionally filtered by set."""
    class_obj = Class.query.get(class_id)
    if not class_obj:
        return jsonify({'error': 'Class not found.'}), 404

    set_id = request.args.get('set_id', type=int)
    query = Flashcard.query.filter_by(class_id=class_id)
    if set_id:
        query = query.filter_by(set_id=set_id)
    flashcards = query.all()

    if not flashcards:
        return jsonify({'error': 'No flashcards to export.'}), 400

    format_choice = request.args.get('format', 'quizlet')
    cards_data = [{'term': c.term, 'definition': c.definition} for c in flashcards]

    if format_choice == 'anki':
        content = export_to_anki(cards_data)
        filename = f"{class_obj.name.replace(' ', '_')}_flashcards_anki.csv"
        mimetype = 'text/csv'
    else:
        content = export_to_quizlet(cards_data)
        filename = f"{class_obj.name.replace(' ', '_')}_flashcards_quizlet.tsv"
        mimetype = 'text/tab-separated-values'

    # Serve directly from memory — no temp files to orphan
    buffer = io.BytesIO(content.encode('utf-8'))
    return send_file(
        buffer,
        as_attachment=True,
        download_name=filename,
        mimetype=mimetype,
    )
