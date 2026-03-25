"""Input model - uploaded materials (PDFs, recordings, notes)."""
from datetime import datetime
from app.extensions import db

class Input(db.Model):
    """
    An uploaded input file for a class.

    Types: 'slides', 'recording', 'notes', 'textbook', 'quiz'
    """
    __tablename__ = 'inputs'

    id = db.Column(db.Integer, primary_key=True)
    class_id = db.Column(db.Integer, db.ForeignKey('classes.id'), nullable=False)
    name = db.Column(db.String(200), nullable=False)
    input_type = db.Column(db.String(50), nullable=False)
    file_path = db.Column(db.String(500))
    raw_text = db.Column(db.Text)
    extraction_method = db.Column(db.String(50))  # 'pypdf', 'tesseract', 'mathpix', 'claude', 'pptx'
    summary = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    class_ = db.relationship('Class', back_populates='inputs')
    flashcards = db.relationship('Flashcard', back_populates='input')

    def __repr__(self):
        return f'<Input {self.id}: {self.name} ({self.input_type})>'
