import os
from pathlib import Path
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE_DIR = Path(__file__).resolve().parent

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'lion-scribe-live-audio-secret-key-2026')
    
    # SQLite Database
    INSTANCE_DIR = BASE_DIR / 'instance'
    INSTANCE_DIR.mkdir(exist_ok=True)
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        'DATABASE_URL', 
        f'sqlite:///{INSTANCE_DIR / "lionscribe.db"}'
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Storage Paths
    STORAGE_DIR = BASE_DIR / 'storage'
    AUDIO_DIR = STORAGE_DIR / 'audio'
    TRANSCRIPT_DIR = STORAGE_DIR / 'transcripts'
    SUMMARY_DIR = STORAGE_DIR / 'summaries'

    for directory in [STORAGE_DIR, AUDIO_DIR, TRANSCRIPT_DIR, SUMMARY_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    # Whisper Settings (Options: 'distil-small.en', 'base.en', 'small.en')
    # Default is 'distil-small.en' - fast and highly accurate English transcription
    WHISPER_MODEL_NAME = os.environ.get('WHISPER_MODEL', 'distil-small.en')

    # Llama 3 Settings (Supports Ollama, Groq, OpenRouter, or OpenAI-compatible endpoint)
    LLAMA_MODEL = os.environ.get('LLAMA_MODEL', 'llama3:8b')
    LLAMA_API_BASE = os.environ.get('LLAMA_API_BASE', 'http://localhost:11434/api')
    LLAMA_API_KEY = os.environ.get('LLAMA_API_KEY', '')
    GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')
