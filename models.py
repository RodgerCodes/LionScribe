import uuid
from datetime import datetime, timezone
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Meeting(db.Model):
    __tablename__ = 'meetings'

    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(36), unique=True, default=lambda: str(uuid.uuid4()), nullable=False)
    title = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text, nullable=True)
    status = db.Column(db.String(50), default='created', nullable=False)  # created, recording, processing, completed, error
    
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    ended_at = db.Column(db.DateTime, nullable=True)
    duration_seconds = db.Column(db.Integer, default=0)

    # File paths (relative to storage folders)
    audio_file = db.Column(db.String(255), nullable=True)          # e.g., <session_id>.mp3
    transcript_file = db.Column(db.String(255), nullable=True)     # e.g., <session_id>.txt
    summary_pdf_file = db.Column(db.String(255), nullable=True)    # e.g., <session_id>_summary.pdf

    # Stored content
    full_transcript = db.Column(db.Text, default="", nullable=True)
    summary_markdown = db.Column(db.Text, default="", nullable=True)

    # Relationships
    segments = db.relationship('TranscriptSegment', backref='meeting', lazy=True, cascade='all, delete-orphan', order_by='TranscriptSegment.sequence')

    def to_dict(self):
        return {
            'id': self.id,
            'session_id': self.session_id,
            'title': self.title,
            'description': self.description,
            'status': self.status,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'ended_at': self.ended_at.isoformat() if self.ended_at else None,
            'duration_seconds': self.duration_seconds,
            'has_audio': bool(self.audio_file),
            'has_transcript': bool(self.transcript_file),
            'has_summary_pdf': bool(self.summary_pdf_file),
            'full_transcript': self.full_transcript or "",
            'summary_markdown': self.summary_markdown or ""
        }

class TranscriptSegment(db.Model):
    __tablename__ = 'transcript_segments'

    id = db.Column(db.Integer, primary_key=True)
    meeting_id = db.Column(db.Integer, db.ForeignKey('meetings.id'), nullable=False)
    sequence = db.Column(db.Integer, nullable=False, default=0)
    speaker = db.Column(db.String(50), default="Speaker", nullable=True)
    text = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    def to_dict(self):
        return {
            'id': self.id,
            'sequence': self.sequence,
            'speaker': self.speaker,
            'text': self.text,
            'created_at': self.created_at.isoformat()
        }
