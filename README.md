# 🦁 LionScribe &mdash; Live Audio Transcription & AI Meeting Notes

LionScribe is a web application built with **Flask**, **SQLite**, and styled using the **Tailwind CSS CDN**. It records live microphone audio during meetings, transcribes speech continuously with **OpenAI Whisper** (base model), exports `.mp3` audio and `.txt` transcripts, and uses **Llama 3** to automatically generate structured meeting notes and a formatted **PDF summary report**.

---

## ✨ Features

- **New Meeting Creation on Launch**:
  - The home page opens directly with a clean "Create a New Meeting" form.
- **Big Start Meeting Button**:
  - Interactive live room with a large, accessible Start button to trigger microphone recording and live speech capture.
- **Live OpenAI Whisper Transcription**:
  - Continuous audio stream transcription using OpenAI Whisper (`base` model &mdash; lightweight, fast, and balanced accuracy).
- **Persistent Storage**:
  - **Audio**: Finalized `.mp3` audio file saved in `storage/audio/`.
  - **Transcripts**: Formatted `.txt` transcript file saved in `storage/transcripts/`.
  - **Database**: SQLite database (`instance/lionscribe.db`) storing meeting metadata, timestamps, durations, and segments.
- **Llama 3 Meeting Notes & PDF Summary**:
  - Triggered immediately on clicking **"End Meeting"**.
  - Summarizes the transcript into Executive Summary, Key Discussion Points, Decisions Made, and Action Items with checkboxes.
  - Generates a styled, downloadable **Meeting Summary PDF** report via ReportLab in `storage/summaries/`.
- **Tailwind CSS CDN UI**:
  - Modern, responsive dark-mode interface with live audio wave animations, timer, audio player, and instant download buttons.

---

## 📁 Project Structure

```
lionscribe/
├── app.py                      # Flask routes, WebSocket/API endpoints, file delivery
├── config.py                   # Configuration for directories, Whisper & Llama 3
├── models.py                   # SQLite models (Meeting, TranscriptSegment)
├── requirements.txt            # Python dependencies
├── .env.example                # Example environment variables
├── .gitignore                  # Git ignore rules
├── services/
│   ├── __init__.py
│   ├── audio_service.py        # Audio chunk processing & MP3 encoding via ffmpeg
│   ├── whisper_service.py      # OpenAI Whisper model loader & .txt export
│   ├── llama_service.py        # Llama 3 notes summarizer (Ollama/Groq/OpenAI/Fallback)
│   └── pdf_service.py          # ReportLab PDF generator for meeting summaries
├── storage/                    # Output directory for saved artifacts
│   ├── audio/                  # Saved .mp3 audio recordings
│   ├── transcripts/            # Saved .txt meeting transcripts
│   └── summaries/              # Saved .pdf meeting summary reports
├── static/
│   └── js/
│       └── meeting.js          # Web audio recording, live streaming & UI updates
└── templates/
    ├── base.html               # Tailwind CSS CDN layout & navigation
    ├── index.html              # "Create a New Meeting" form & past meeting archive
    ├── meeting.html            # Live room with Big Start Button & live Whisper stream
    └── summary.html            # Completed meeting notes, audio player & download links
```

---

## 🚀 Getting Started

### 1. Requirements
- Python 3.10+ (Virtual environment available at `~/scribe/bin`)
- `ffmpeg` (installed on system)

### 2. Install Dependencies
```bash
~/scribe/bin/pip install -r requirements.txt
```

### 3. Run the Application
```bash
~/scribe/bin/python app.py
```
Open your browser and navigate to: **`http://localhost:5000`**

---

## ⚙️ AI Models Configuration

LionScribe uses balanced, accessible models:

| Component | Default Model | Customization |
| :--- | :--- | :--- |
| **Speech Transcription** | **OpenAI Whisper `base`** | Set `WHISPER_MODEL=small` or `tiny` in `.env` |
| **Meeting Notes Summary** | **Llama 3 (8B)** | Supports Ollama (`llama3:8b`), Groq (`GROQ_API_KEY`), or built-in fallback |

To configure remote or local Llama 3 endpoints, copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
And adjust `LLAMA_MODEL`, `LLAMA_API_BASE`, or `GROQ_API_KEY`.
