"""FlashcardSet model - groups of generated flashcards."""
from datetime import datetime
from app.extensions import db


class FlashcardSet(db.Model):
    """
    A named set of flashcards, auto-created per generation.

    Each time the user generates flashcards, a new set is created
    with the topic as the name.
    """
    __tablename__ = 'flashcard_sets'

    id = db.Column(db.Integer, primary_key=True)
    class_id = db.Column(db.Integer, db.ForeignKey('classes.id'), nullable=False)
    name = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    class_ = db.relationship('Class', back_populates='flashcard_sets')
    flashcards = db.relationship('Flashcard', back_populates='flashcard_set',
                                 cascade='all, delete-orphan')

    def __repr__(self):
        return f'<FlashcardSet {self.id}: {self.name}>'

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'card_count': len(self.flashcards),
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }
