import os
from datetime import datetime, timezone
from pathlib import Path
from flask import (
    Flask, render_template, request, redirect, url_for, 
    jsonify, send_file, flash, abort
)
import json
from flask_sock import Sock
from config import Config
from models import db, Meeting, TranscriptSegment
from services.audio_service import AudioService
from services.whisper_service import WhisperService
from services.llama_service import LlamaService
from services.pdf_service import PDFService

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    sock = Sock(app)

    # Initialize extensions
    db.init_app(app)

    with app.app_context():
        db.create_all()

    # -------------------------------------------------------------------------
    # Web Page Routes
    # -------------------------------------------------------------------------
    @app.route('/')
    def index():
        """Home page: 'Create a new meeting' + list of past meetings."""
        past_meetings = Meeting.query.order_by(Meeting.created_at.desc()).all()
        return render_template('index.html', meetings=past_meetings)

    @app.route('/meetings/create', methods=['POST'])
    def create_meeting():
        """Creates a new meeting record and opens the live room."""
        title = request.form.get('title', '').strip()
        description = request.form.get('description', '').strip()

        if not title:
            title = f"Meeting on {datetime.now().strftime('%b %d, %Y - %H:%M')}"

        new_meeting = Meeting(
            title=title,
            description=description,
            status='created'
        )
        db.session.add(new_meeting)
        db.session.commit()

        flash(f'Meeting "{title}" created! Ready to start recording.', 'success')
        return redirect(url_for('meeting_room', session_id=new_meeting.session_id))

    @app.route('/meetings/<session_id>')
    def meeting_room(session_id):
        """Live meeting room with big Start Button, Whisper stream, and End button."""
        meeting = Meeting.query.filter_by(session_id=session_id).first_or_404()
        if meeting.status == 'completed':
            return redirect(url_for('meeting_summary', session_id=session_id))
        return render_template('meeting.html', meeting=meeting)

    @app.route('/meetings/<session_id>/summary')
    def meeting_summary(session_id):
        """Meeting summary page showing audio player, .txt transcript, and PDF download."""
        meeting = Meeting.query.filter_by(session_id=session_id).first_or_404()
        return render_template('summary.html', meeting=meeting)

    @app.route('/meetings/<session_id>/delete', methods=['POST'])
    def delete_meeting(session_id):
        """Deletes a meeting and its stored files."""
        meeting = Meeting.query.filter_by(session_id=session_id).first_or_404()
        
        # Cleanup files if they exist
        for path_attr, base_folder in [
            (meeting.audio_file, Config.AUDIO_DIR),
            (meeting.transcript_file, Config.TRANSCRIPT_DIR),
            (meeting.summary_pdf_file, Config.SUMMARY_DIR)
        ]:
            if path_attr:
                file_path = base_folder / path_attr
                if file_path.exists():
                    try:
                        file_path.unlink()
                    except Exception:
                        pass

        db.session.delete(meeting)
        db.session.commit()
        flash('Meeting record and files deleted.', 'info')
        return redirect(url_for('index'))

    # -------------------------------------------------------------------------
    # Real-Time WebSocket & API Endpoints
    # -------------------------------------------------------------------------
    @sock.route('/ws/transcribe/<session_id>')
    def ws_transcribe(ws, session_id):
        """
        High-performance persistent WebSocket for streaming raw audio chunks.
        Transcribes immediately with Whisper INT8 and streams text back with conversational context.
        """
        meeting = Meeting.query.filter_by(session_id=session_id).first()
        chunk_counter = 0

        while True:
            try:
                data = ws.receive()
                if data is None:
                    break

                if isinstance(data, (bytes, bytearray)) and len(data) > 44:
                    chunk_counter += 1
                    # Save chunk to disk for audio archive
                    AudioService.save_chunk(session_id, chunk_counter, data, ext="wav")
                    
                    # Transcribe chunk with conversational context carryover
                    prompt_context = meeting.full_transcript if meeting else ""
                    text = WhisperService.transcribe_bytes(
                        data, 
                        model_name=Config.WHISPER_MODEL_NAME,
                        prompt_context=prompt_context
                    )
                    if text:
                        if meeting:
                            current_full = meeting.full_transcript or ""
                            meeting.full_transcript = f"{current_full} {text}".strip() if current_full else text
                            segment = TranscriptSegment(meeting_id=meeting.id, sequence=chunk_counter, text=text)
                            db.session.add(segment)
                            db.session.commit()

                        ws.send(json.dumps({
                            'chunk_index': chunk_counter,
                            'chunk_text': text,
                            'full_transcript': meeting.full_transcript if meeting else text
                        }))
            except Exception as e:
                print(f"[WebSocket] Error in ws_transcribe: {e}")
                break

    @app.route('/api/meetings/<session_id>/start', methods=['POST'])
    def api_start_meeting(session_id):
        """Mark meeting status as recording."""
        meeting = Meeting.query.filter_by(session_id=session_id).first_or_404()
        meeting.status = 'recording'
        db.session.commit()
        return jsonify({'status': 'recording', 'session_id': session_id})

    @app.route('/api/meetings/<session_id>/transcribe-chunk', methods=['POST'])
    def api_transcribe_chunk(session_id):
        """
        Receives an audio chunk from the browser, converts it,
        transcribes it with Whisper, and stores the segment.
        """
        meeting = Meeting.query.filter_by(session_id=session_id).first_or_404()
        
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file uploaded'}), 400

        audio_file = request.files['audio']
        chunk_index = int(request.form.get('chunk_index', 0))
        mime_type = request.form.get('mime_type', 'audio/webm')

        ext = 'webm'
        if 'ogg' in mime_type:
            ext = 'ogg'
        elif 'wav' in mime_type:
            ext = 'wav'

        # 1. Save raw chunk
        raw_chunk_path = AudioService.save_chunk(
            session_id=session_id,
            chunk_index=chunk_index,
            audio_data=audio_file.read(),
            ext=ext
        )

        # 2. Convert to 16kHz mono WAV for Whisper
        wav_path = AudioService.convert_to_wav_for_whisper(raw_chunk_path)

        # 3. Transcribe with Whisper using conversational context
        prompt_context = meeting.full_transcript if meeting else ""
        with open(wav_path, 'rb') as f:
            wav_bytes = f.read()
        chunk_text = WhisperService.transcribe_bytes(
            wav_bytes,
            model_name=Config.WHISPER_MODEL_NAME,
            prompt_context=prompt_context
        )

        # 4. Save segment if meaningful text transcribed
        if chunk_text:
            segment = TranscriptSegment(
                meeting_id=meeting.id,
                sequence=chunk_index,
                text=chunk_text
            )
            db.session.add(segment)

            # Append to meeting's full transcript
            current_full = meeting.full_transcript or ""
            if current_full:
                meeting.full_transcript = f"{current_full} {chunk_text}".strip()
            else:
                meeting.full_transcript = chunk_text

            db.session.commit()

        return jsonify({
            'chunk_index': chunk_index,
            'chunk_text': chunk_text,
            'full_transcript': meeting.full_transcript or ""
        })

    @app.route('/api/meetings/<session_id>/end', methods=['POST'])
    def api_end_meeting(session_id):
        """
        Finalizes meeting:
        1. Encodes final .mp3 audio file
        2. Performs sentence-level full audio transcript alignment
        3. Writes full .txt transcript file
        4. Generates Llama 3 notes summary
        5. Generates notes summary PDF
        6. Marks meeting completed in SQLite
        """
        meeting = Meeting.query.filter_by(session_id=session_id).first_or_404()
        meeting.status = 'processing'
        db.session.commit()

        # 1. Check if user sent a complete single audio recording file in the request
        single_audio_file = None
        if 'full_audio' in request.files:
            uploaded_audio = request.files['full_audio']
            tmp_full = Config.STORAGE_DIR / f"tmp_{session_id}.webm"
            uploaded_audio.save(tmp_full)
            single_audio_file = tmp_full

        # Duration calculation
        duration_seconds = int(request.form.get('duration_seconds', 0))
        if duration_seconds <= 0 and meeting.created_at:
            duration_seconds = int((datetime.now(timezone.utc) - meeting.created_at.replace(tzinfo=timezone.utc)).total_seconds())

        meeting.duration_seconds = duration_seconds
        meeting.ended_at = datetime.now(timezone.utc)

        # 2. Finalize MP3 file
        mp3_filename = f"{session_id}.mp3"
        mp3_path = Config.AUDIO_DIR / mp3_filename
        AudioService.finalize_recording_to_mp3(
            session_id=session_id,
            output_mp3_path=mp3_path,
            single_audio_file=single_audio_file
        )
        if mp3_path.exists():
            meeting.audio_file = mp3_filename

        # Clean up temporary full audio file if used
        if single_audio_file and single_audio_file.exists():
            try:
                single_audio_file.unlink()
            except Exception:
                pass

        # 3. High-Accuracy Full Continuous Audio Alignment Pass
        # Transcribes the full recorded MP3 for perfect conversational coherence and sentence timestamps
        full_aligned_text = ""
        if mp3_path.exists() and mp3_path.stat().st_size > 4000:
            try:
                aligned_res = WhisperService.transcribe_audio_file(mp3_path, model_name=Config.WHISPER_MODEL_NAME)
                full_aligned_text = aligned_res.get('text', '').strip()
            except Exception as e:
                print(f"[app] Full audio alignment notice: {e}")

        # If full-pass transcribed successfully, use it; otherwise fallback to client/live transcript
        if full_aligned_text:
            meeting.full_transcript = full_aligned_text
        else:
            final_transcript_client = request.form.get('final_transcript', '').strip()
            if final_transcript_client and len(final_transcript_client) > len(meeting.full_transcript or ""):
                meeting.full_transcript = final_transcript_client

        # 4. Save .txt transcript file
        txt_filename = f"{session_id}.txt"
        txt_path = Config.TRANSCRIPT_DIR / txt_filename
        created_str = meeting.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')
        WhisperService.save_transcript_file(
            meeting_title=meeting.title,
            created_at_str=created_str,
            transcript_text=meeting.full_transcript or "[No speech detected]",
            output_txt_path=txt_path,
            model_name=Config.WHISPER_MODEL_NAME
        )
        meeting.transcript_file = txt_filename

        # 5. Generate Llama 3 Notes Summary
        summary_markdown = LlamaService.generate_meeting_summary(
            meeting_title=meeting.title,
            transcript_text=meeting.full_transcript or ""
        )
        meeting.summary_markdown = summary_markdown

        # 6. Generate Summary PDF
        pdf_filename = f"{session_id}_summary.pdf"
        pdf_path = Config.SUMMARY_DIR / pdf_filename
        mins, secs = divmod(duration_seconds, 60)
        duration_formatted = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
        
        PDFService.generate_meeting_pdf(
            meeting_title=meeting.title,
            created_at_str=created_str,
            duration_str=duration_formatted,
            summary_markdown=summary_markdown,
            transcript_text=meeting.full_transcript or "",
            output_pdf_path=pdf_path
        )
        meeting.summary_pdf_file = pdf_filename

        # 7. Update status to completed
        meeting.status = 'completed'
        db.session.commit()

        return jsonify({
            'success': True,
            'redirect_url': url_for('meeting_summary', session_id=session_id)
        })

    # -------------------------------------------------------------------------
    # File Download / Streaming Endpoints
    # -------------------------------------------------------------------------
    @app.route('/download/audio/<session_id>')
    def download_audio(session_id):
        """Download or stream recorded MP3 audio."""
        meeting = Meeting.query.filter_by(session_id=session_id).first_or_404()
        if not meeting.audio_file:
            abort(404, "Audio file not available")
        path = Config.AUDIO_DIR / meeting.audio_file
        if not path.exists():
            abort(404, "Audio file does not exist on disk")
        return send_file(path, mimetype='audio/mpeg', as_attachment=request.args.get('download') == '1', download_name=f"{meeting.title}.mp3")

    @app.route('/download/transcript/<session_id>')
    def download_transcript(session_id):
        """Download .txt transcript file."""
        meeting = Meeting.query.filter_by(session_id=session_id).first_or_404()
        if not meeting.transcript_file:
            abort(404, "Transcript file not available")
        path = Config.TRANSCRIPT_DIR / meeting.transcript_file
        if not path.exists():
            abort(404, "Transcript file does not exist on disk")
        return send_file(path, mimetype='text/plain', as_attachment=True, download_name=f"{meeting.title}_transcript.txt")

    @app.route('/download/pdf/<session_id>')
    def download_pdf(session_id):
        """Download or view Llama 3 notes summary PDF."""
        meeting = Meeting.query.filter_by(session_id=session_id).first_or_404()
        if not meeting.summary_pdf_file:
            abort(404, "Summary PDF not available")
        path = Config.SUMMARY_DIR / meeting.summary_pdf_file
        if not path.exists():
            abort(404, "Summary PDF does not exist on disk")
        return send_file(path, mimetype='application/pdf', as_attachment=request.args.get('download') == '1', download_name=f"{meeting.title}_notes_summary.pdf")

    return app

app = create_app()

if __name__ == '__main__':
    import socket
    
    # Get local IP address dynamically
    def get_local_ip():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    local_ip = get_local_ip()
    cert_file = Config.INSTANCE_DIR / 'certs' / 'cert.pem'
    key_file = Config.INSTANCE_DIR / 'certs' / 'key.pem'

    use_ssl = (os.environ.get('USE_SSL', '1') == '1') and cert_file.exists() and key_file.exists()
    ssl_context = (str(cert_file), str(key_file)) if use_ssl else None
    scheme = "https" if ssl_context else "http"

    print("\n" + "=" * 60)
    print(" 🦁 LionScribe Live Server Running")
    print("=" * 60)
    print(f" 👉 Local PC:   {scheme}://localhost:5000")
    print(f" 👉 Tablet/LAN: {scheme}://{local_ip}:5000")
    if ssl_context:
        print(" 🔒 HTTPS Active: Enables microphone access on tablet browsers.")
        print("    (If your tablet shows a certificate warning, tap 'Advanced' -> 'Proceed').")
    else:
        print(" ℹ️ HTTP Mode: For tablet mic access, consider enabling HTTPS certs.")
    print("=" * 60 + "\n")

    app.run(
        debug=False,
        host='0.0.0.0',
        port=5000,
        threaded=True,
        ssl_context=ssl_context
    )
