import os
import io
import wave
import subprocess
import threading
import urllib.request
import json
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np

class WhisperService:
    _loaded_models: Dict[str, Any] = {}
    _lock = threading.Lock()

    @classmethod
    def get_model(cls, model_name: str = "distil-large-v3"):
        """
        Loads faster-whisper model (e.g. distil-large-v3, distil-medium.en, small.en)
        with INT8 quantization on CPU for high accuracy and fast inference.
        """
        if model_name in cls._loaded_models:
            return cls._loaded_models[model_name]

        with cls._lock:
            if model_name in cls._loaded_models:
                return cls._loaded_models[model_name]

            cpu_threads = min(os.cpu_count() or 4, 8)
            candidates = [model_name, "distil-large-v3", "distil-medium.en", "small.en", "base.en"]
            # Deduplicate while preserving order
            seen = set()
            ordered_candidates = [c for c in candidates if not (c in seen or seen.add(c))]

            for candidate in ordered_candidates:
                try:
                    from faster_whisper import WhisperModel
                    print(f"[WhisperService] Loading faster-whisper model '{candidate}' (INT8 on {cpu_threads} CPU threads)...")
                    model = WhisperModel(candidate, device="cpu", compute_type="int8", cpu_threads=cpu_threads)
                    cls._loaded_models[candidate] = model
                    cls._loaded_models[model_name] = model
                    print(f"[WhisperService] Successfully loaded faster-whisper model '{candidate}'.")
                    return model
                except Exception as e:
                    print(f"[WhisperService] Could not load candidate '{candidate}': {e}")

            # OpenAI Whisper fallback
            try:
                import whisper
                print("[WhisperService] Falling back to standard openai-whisper (base)...")
                model = whisper.load_model('base')
                cls._loaded_models[model_name] = model
                return model
            except Exception as e2:
                print(f"[WhisperService] Critical: could not load any whisper model: {e2}")
                return None

    @classmethod
    def decode_wav_to_numpy(cls, wav_data: bytes) -> Optional[np.ndarray]:
        """
        Decodes in-memory WAV bytes directly into a 16kHz float32 NumPy array.
        """
        try:
            with wave.open(io.BytesIO(wav_data), 'rb') as wf:
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                framerate = wf.getframerate()
                n_frames = wf.getnframes()
                audio_bytes = wf.readframes(n_frames)

                if sampwidth == 2:
                    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                elif sampwidth == 4:
                    audio_np = np.frombuffer(audio_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
                else:
                    audio_np = np.frombuffer(audio_bytes, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0

                if n_channels > 1:
                    audio_np = audio_np.reshape(-1, n_channels).mean(axis=1)

                if framerate != 16000 and len(audio_np) > 0:
                    indices = np.linspace(0, len(audio_np) - 1, int(len(audio_np) * 16000 / framerate))
                    audio_np = np.interp(indices, np.arange(len(audio_np)), audio_np)

                if len(audio_np) > 0:
                    return audio_np
        except Exception as e:
            print(f"[WhisperService] Error decoding WAV bytes: {e}")
        return None

    @classmethod
    def transcribe_bytes(cls, audio_bytes: bytes, model_name: str = "distil-large-v3", prompt_context: str = "") -> str:
        """
        High-accuracy conversational English transcription:
        1. Checks for GROQ_API_KEY (Whisper-Large-v3-turbo cloud API)
        2. Uses local faster-whisper model with INT8 quantization, conversational VAD, and context prompt
        """
        # 1. Cloud Groq Whisper (if API key provided)
        groq_key = os.environ.get('GROQ_API_KEY', '').strip()
        if groq_key and len(audio_bytes) > 4000:
            try:
                boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
                prompt_field = ""
                if prompt_context and prompt_context.strip():
                    recent = prompt_context.strip()[-200:]
                    prompt_field = (
                        f'--{boundary}\r\n'
                        f'Content-Disposition: form-data; name="prompt"\r\n\r\n{recent}\r\n'
                    )

                body = (
                    f'--{boundary}\r\n'
                    f'Content-Disposition: form-data; name="model"\r\n\r\nwhisper-large-v3-turbo\r\n'
                    f'--{boundary}\r\n'
                    f'Content-Disposition: form-data; name="language"\r\n\r\nen\r\n'
                    f'{prompt_field}'
                    f'--{boundary}\r\n'
                    f'Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n'
                    f'Content-Type: audio/wav\r\n\r\n'
                ).encode('utf-8') + audio_bytes + f'\r\n--{boundary}--\r\n'.encode('utf-8')

                req = urllib.request.Request(
                    "https://api.groq.com/openai/v1/audio/transcriptions",
                    data=body,
                    headers={
                        "Authorization": f"Bearer {groq_key}",
                        "Content-Type": f"multipart/form-data; boundary={boundary}"
                    }
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    if resp.status == 200:
                        data = json.loads(resp.read().decode('utf-8'))
                        text = data.get('text', '').strip()
                        if text:
                            return text
            except Exception as e:
                print(f"[WhisperService] Groq whisper notice: {e}")

        # 2. Local Faster-Whisper
        audio_np = cls.decode_wav_to_numpy(audio_bytes)
        if audio_np is None or len(audio_np) < 4800:  # Ignore audio under 0.3s
            return ""

        # Energy check: Skip pure digital silence / flat background noise
        rms = np.sqrt(np.mean(audio_np ** 2))
        if rms < 0.005:
            return ""

        model = cls.get_model(model_name)
        if model is None:
            return ""

        # Build context-aware prompt for conversational continuity
        initial_prompt = "High accuracy English meeting transcription with complete sentences and correct punctuation."
        if prompt_context and prompt_context.strip():
            recent_context = prompt_context.strip()[-200:]
            initial_prompt = f"{initial_prompt} Previous context: {recent_context}"

        try:
            if hasattr(model, 'transcribe') and 'WhisperModel' in str(type(model)):
                segments, _ = model.transcribe(
                    audio_np,
                    beam_size=1,
                    best_of=1,
                    temperature=0.0,
                    language='en',
                    without_timestamps=True,
                    vad_filter=True,
                    vad_parameters=dict(
                        min_silence_duration_ms=400,
                        speech_pad_ms=150
                    ),
                    initial_prompt=initial_prompt
                )
                text = " ".join([seg.text.strip() for seg in segments if seg.text.strip()]).strip()
                return text
            else:
                res = model.transcribe(audio_np, fp16=False, language='en')
                return res.get('text', '').strip()
        except Exception as e:
            print(f"[WhisperService] Transcription error: {e}")
            return ""

    @classmethod
    def transcribe_audio_file(cls, audio_path: Path, model_name: str = "distil-large-v3") -> Dict[str, Any]:
        """
        Transcribes a complete audio file with conversational sentence-level timestamps.
        Returns a dict containing:
        - "text": Clean formatted transcript with timestamps
        - "raw_text": Un-timestamped text
        - "segments": List of segment objects with start, end, timestamp, text
        """
        if not audio_path.exists() or audio_path.stat().st_size == 0:
            return {"text": "", "raw_text": "", "segments": []}

        # First try Groq if key is available and file < 25MB
        groq_key = os.environ.get('GROQ_API_KEY', '').strip()
        if groq_key and audio_path.stat().st_size < 25 * 1024 * 1024:
            try:
                with open(audio_path, 'rb') as f:
                    file_data = f.read()
                text = cls.transcribe_bytes(file_data, model_name=model_name)
                if text:
                    return {
                        "text": text,
                        "raw_text": text,
                        "segments": [{"start": 0, "end": 0, "timestamp": "[00:00]", "text": text}]
                    }
            except Exception as e:
                print(f"[WhisperService] Groq file transcribe error: {e}")

        model = cls.get_model(model_name)
        if model is None:
            return {"text": "", "raw_text": "", "segments": []}

        try:
            if hasattr(model, 'transcribe') and 'WhisperModel' in str(type(model)):
                segments, info = model.transcribe(
                    str(audio_path),
                    beam_size=1,
                    best_of=1,
                    temperature=0.0,
                    language='en',
                    vad_filter=True,
                    vad_parameters=dict(
                        min_silence_duration_ms=500,
                        speech_pad_ms=200
                    ),
                    initial_prompt="High accuracy English meeting transcription with complete conversational sentences and correct punctuation."
                )

                formatted_lines = []
                raw_lines = []
                parsed_segments = []

                for seg in segments:
                    seg_text = seg.text.strip()
                    if not seg_text:
                        continue

                    # Format timestamp [MM:SS] or [HH:MM:SS]
                    total_secs = int(seg.start)
                    mins, secs = divmod(total_secs, 60)
                    hrs, mins = divmod(mins, 60)
                    if hrs > 0:
                        ts_str = f"[{hrs:02d}:{mins:02d}:{secs:02d}]"
                    else:
                        ts_str = f"[{mins:02d}:{secs:02d}]"

                    formatted_lines.append(f"{ts_str} {seg_text}")
                    raw_lines.append(seg_text)
                    parsed_segments.append({
                        "start": round(seg.start, 2),
                        "end": round(seg.end, 2),
                        "timestamp": ts_str,
                        "text": seg_text
                    })

                full_formatted = "\n\n".join(formatted_lines)
                raw_text = " ".join(raw_lines)
                return {
                    "text": full_formatted,
                    "raw_text": raw_text,
                    "segments": parsed_segments
                }
            else:
                res = model.transcribe(str(audio_path), fp16=False, language='en')
                text = res.get('text', '').strip()
                return {"text": text, "raw_text": text, "segments": []}
        except Exception as e:
            print(f"[WhisperService] Audio file transcription error: {e}")
            return {"text": "", "raw_text": "", "segments": []}

    @classmethod
    def transcribe_audio(cls, audio_path: Path, model_name: str = "distil-large-v3") -> Dict[str, Any]:
        """
        Backward compatibility wrapper.
        """
        return cls.transcribe_audio_file(audio_path, model_name=model_name)

    @classmethod
    def save_transcript_file(cls, meeting_title: str, created_at_str: str, transcript_text: str, output_txt_path: Path, model_name: str = "distil-large-v3") -> Path:
        """
        Saves the complete meeting transcription to a clean .txt file with metadata.
        """
        output_txt_path.parent.mkdir(parents=True, exist_ok=True)
        content = (
            f"===========================================================\n"
            f" LionScribe Meeting Transcript\n"
            f"===========================================================\n"
            f" Title     : {meeting_title}\n"
            f" Date      : {created_at_str}\n"
            f" Engine    : Faster-Whisper ({model_name} / INT8)\n"
            f"===========================================================\n\n"
            f"--- TRANSCRIPT CONTENT ---\n\n"
            f"{transcript_text.strip()}\n\n"
            f"===========================================================\n"
            f" End of Transcript\n"
            f"===========================================================\n"
        )
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write(content)
        return output_txt_path
