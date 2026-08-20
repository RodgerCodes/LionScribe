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
    _faster_model = None
    _openai_model = None
    _lock = threading.Lock()

    @classmethod
    def get_model(cls, model_name: str = "distil-small.en"):
        """
        Loads faster-whisper (distil-small.en / small.en) with INT8 quantization.
        Distil-Whisper provides high accuracy with 6x faster inference speed on CPU.
        """
        if cls._faster_model is None:
            with cls._lock:
                if cls._faster_model is None:
                    try:
                        from faster_whisper import WhisperModel
                        print(f"[WhisperService] Loading faster-whisper ({model_name}, int8 on CPU)...")
                        cls._faster_model = WhisperModel(model_name, device="cpu", compute_type="int8", cpu_threads=8)
                        print(f"[WhisperService] faster-whisper ({model_name}) loaded successfully.")
                        return cls._faster_model
                    except Exception as e:
                        print(f"[WhisperService] distil-small.en failed ({e}), trying base.en / standard whisper...")
                        try:
                            from faster_whisper import WhisperModel
                            cls._faster_model = WhisperModel('base.en', device="cpu", compute_type="int8", cpu_threads=8)
                            return cls._faster_model
                        except Exception:
                            try:
                                import whisper
                                cls._openai_model = whisper.load_model('base')
                                return cls._openai_model
                            except Exception as e2:
                                print(f"[WhisperService] Could not load whisper: {e2}")
        return cls._faster_model or cls._openai_model

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
        except Exception:
            pass
        return None

    @classmethod
    def transcribe_bytes(cls, audio_bytes: bytes, model_name: str = "distil-small.en") -> str:
        """
        High-accuracy English transcription:
        1. Checks for GROQ_API_KEY (ultra-fast Whisper-Large-v3-turbo, ~100ms)
        2. Falls back to local Distil-Whisper INT8 on CPU with Voice Activity Filtering
        """
        # 1. Cloud Groq Whisper (if API key provided)
        groq_key = os.environ.get('GROQ_API_KEY', '').strip()
        if groq_key and len(audio_bytes) > 4000:
            try:
                boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
                body = (
                    f'--{boundary}\r\n'
                    f'Content-Disposition: form-data; name="model"\r\n\r\nwhisper-large-v3-turbo\r\n'
                    f'--{boundary}\r\n'
                    f'Content-Disposition: form-data; name="language"\r\n\r\nen\r\n'
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

        # 2. Local Faster-Whisper (Distil-Small / Base)
        audio_np = cls.decode_wav_to_numpy(audio_bytes)
        if audio_np is None or len(audio_np) < 4000: # Ignore noise under 0.25s
            return ""

        model = cls.get_model(model_name)
        if model is None:
            return ""

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
                        min_silence_duration_ms=300,
                        speech_pad_ms=100
                    ),
                    initial_prompt="High accuracy English meeting transcription with complete sentences and correct punctuation."
                )
                text = " ".join([seg.text for seg in segments]).strip()
                return text
            else:
                res = model.transcribe(audio_np, fp16=False, language='en')
                return res.get('text', '').strip()
        except Exception as e:
            print(f"[WhisperService] Chunk error: {e}")
            return ""

    @classmethod
    def transcribe_audio(cls, audio_path: Path, model_name: str = "distil-small.en") -> Dict[str, Any]:
        """
        Transcribes audio from a file path.
        """
        try:
            with open(audio_path, 'rb') as f:
                data = f.read()
            text = cls.transcribe_bytes(data, model_name=model_name)
            return {"text": text, "segments": [], "language": "en"}
        except Exception as e:
            print(f"[WhisperService] File transcription error: {e}")
            return {"text": "", "segments": [], "language": "en"}

    @classmethod
    def save_transcript_file(cls, meeting_title: str, created_at_str: str, transcript_text: str, output_txt_path: Path) -> Path:
        """
        Saves the complete meeting transcription to a clean .txt file.
        """
        output_txt_path.parent.mkdir(parents=True, exist_ok=True)
        content = (
            f"===========================================================\n"
            f" LionScribe Meeting Transcript\n"
            f"===========================================================\n"
            f" Title     : {meeting_title}\n"
            f" Date      : {created_at_str}\n"
            f" Engine    : Distil-Whisper (distil-small.en / INT8)\n"
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
