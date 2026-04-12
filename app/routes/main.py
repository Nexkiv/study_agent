"""
Page routes for StudyAgent.

Serves the main SPA-like page and HTMX partials.
"""
from flask import Blueprint, render_template, request
from app.extensions import db
from app.models import Class, Input, Flashcard, ChatMessage

main_bp = Blueprint('main', __name__)


@main_bp.route('/')
def index():
    """Main page — single page with tabs for Materials, Chat, Flashcards."""
    classes = Class.query.order_by(Class.name).all()
    # Add file counts for display
    class_data = []
    for cls in classes:
        file_count = Input.query.filter_by(class_id=cls.id).count()
        class_data.append({'id': cls.id, 'name': cls.name, 'file_count': file_count})
    return render_template('index.html', classes=class_data)


@main_bp.route('/partials/materials/<int:class_id>')
def materials_partial(class_id):
    """HTMX partial: file list and upload form for a class."""
    class_obj = Class.query.get_or_404(class_id)
    inputs = Input.query.filter_by(class_id=class_id).order_by(Input.created_at.desc()).all()
    return render_template('partials/_file_list.html', files=inputs, class_obj=class_obj)


@main_bp.route('/partials/chat/<int:class_id>')
def chat_partial(class_id):
    """HTMX partial: chat message history for a class."""
    messages = ChatMessage.query.filter_by(class_id=class_id).order_by(ChatMessage.created_at).all()
    return render_template('partials/_chat_messages.html', messages=messages)


@main_bp.route('/partials/flashcards/<int:class_id>')
def flashcards_partial(class_id):
    """HTMX partial: flashcard table for a class."""
    flashcards = Flashcard.query.filter_by(class_id=class_id).order_by(Flashcard.created_at.desc()).all()
    return render_template('partials/_flashcard_table.html', flashcards=flashcards)
