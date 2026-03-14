"""ChatMessage model - conversation history per class."""
from datetime import datetime
from app.extensions import db

class ChatMessage(db.Model):
    """
    A chat message in the RAG conversation.

    Roles: 'user', 'assistant'
    """
    __tablename__ = 'chat_messages'

    id = db.Column(db.Integer, primary_key=True)
    class_id = db.Column(db.Integer, db.ForeignKey('classes.id'), nullable=False)
    role = db.Column(db.String(20), nullable=False)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    class_ = db.relationship('Class', back_populates='chat_messages')

    __table_args__ = (
        db.CheckConstraint("role IN ('user', 'assistant')", name='valid_role'),
    )

    def __repr__(self):
        return f'<ChatMessage {self.id}: {self.role}>'

    def to_dict(self):
        """Convert to dict for agent history."""
        return {
            'role': self.role,
            'content': self.content
        }
