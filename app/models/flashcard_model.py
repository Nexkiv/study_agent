"""Flashcard model - generated study cards."""
from datetime import datetime
from app.extensions import db

class Flashcard(db.Model):
    """
    A flashcard with term/definition.

    Art history cards include: artist, title, date, period, medium, style.
    """
    __tablename__ = 'flashcards'

    id = db.Column(db.Integer, primary_key=True)
    class_id = db.Column(db.Integer, db.ForeignKey('classes.id'), nullable=False)
    input_id = db.Column(db.Integer, db.ForeignKey('inputs.id'), nullable=True)

    term = db.Column(db.String(500), nullable=False)
    definition = db.Column(db.Text, nullable=False)
    image_url = db.Column(db.String(500))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    class_ = db.relationship('Class', back_populates='flashcards')
    input = db.relationship('Input', back_populates='flashcards')

    def __repr__(self):
        return f'<Flashcard {self.id}: {self.term[:30]}...>'

    def to_dict(self):
        """Convert to dict for Gradio display."""
        return {
            'id': self.id,
            'term': self.term,
            'definition': self.definition,
            'image_url': self.image_url
        }
