"""Quiz model - generated quiz questions (post-MVP)."""
from datetime import datetime
import json
from app.extensions import db

class Quiz(db.Model):
    """
    A quiz question with answer and explanation.

    Supports multiple choice (options as JSON array) and short answer.
    """
    __tablename__ = 'quizzes'

    id = db.Column(db.Integer, primary_key=True)
    class_id = db.Column(db.Integer, db.ForeignKey('classes.id'), nullable=False)

    question = db.Column(db.Text, nullable=False)
    options = db.Column(db.Text)  # JSON array for MC
    answer = db.Column(db.Text, nullable=False)
    explanation = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    class_ = db.relationship('Class', back_populates='quizzes')

    def __repr__(self):
        return f'<Quiz {self.id}: {self.question[:30]}...>'

    @property
    def options_list(self):
        """Parse JSON options to Python list."""
        if self.options:
            return json.loads(self.options)
        return None
