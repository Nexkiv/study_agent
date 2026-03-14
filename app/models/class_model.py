"""Class model - top-level container for course materials."""
from datetime import datetime
from app.extensions import db

class Class(db.Model):
    """
    A class/course container for materials and study aids.

    Example: "Art History 101", "Renaissance Art"
    """
    __tablename__ = 'classes'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    inputs = db.relationship('Input', back_populates='class_',
                            cascade='all, delete-orphan')
    flashcards = db.relationship('Flashcard', back_populates='class_',
                                cascade='all, delete-orphan')
    quizzes = db.relationship('Quiz', back_populates='class_',
                             cascade='all, delete-orphan')
    chat_messages = db.relationship('ChatMessage', back_populates='class_',
                                   cascade='all, delete-orphan')

    def __repr__(self):
        return f'<Class {self.id}: {self.name}>'
