"""
SQLAlchemy ORM models for StudyAgent.

Database schema:
- classes: Top-level class containers
- inputs: Uploaded materials (PDFs, recordings, etc.)
- flashcards: Generated study cards
- quizzes: Generated quiz questions (post-MVP)
- chat_messages: Conversation history per class
"""
from app.models.class_model import Class
from app.models.input_model import Input
from app.models.flashcard_model import Flashcard
from app.models.quiz_model import Quiz
from app.models.chat_model import ChatMessage

__all__ = [
    'Class',
    'Input',
    'Flashcard',
    'Quiz',
    'ChatMessage'
]
