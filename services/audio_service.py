import os
import subprocess
from pathlib import Path
import shutil

class AudioService:
    @staticmethod
    def get_chunk_dir(session_id: str) -> Path:
        base_dir = Path(__file__).resolve().parent.parent / 'instance' / 'temp_audio' / session_id
        base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir

    @classmethod
    def save_chunk(cls, session_id: str, chunk_index: int, audio_data: bytes, ext: str = "wav") -> Path:
        chunk_dir = cls.get_chunk_dir(session_id)
        raw_file = chunk_dir / f"chunk_{chunk_index:05d}.{ext}"
        with open(raw_file, "wb") as f:
            f.write(audio_data)
        return raw_file

    @classmethod
    def convert_to_wav_for_whisper(cls, input_audio_path: Path) -> Path:
        """
        Converts input audio file to 16kHz mono WAV for Whisper.
        If already a valid WAV, returns it immediately.
        """
        if input_audio_path.suffix.lower() == '.wav':
            return input_audio_path

        wav_path = input_audio_path.with_suffix('.whisper.wav')
        cmd = [
            'ffmpeg', '-y',
            '-loglevel', 'error',
            '-i', str(input_audio_path),
            '-ar', '16000',
            '-ac', '1',
            '-c:a', 'pcm_s16le',
            str(wav_path)
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            return input_audio_path
        return wav_path

    @classmethod
    def finalize_recording_to_mp3(cls, session_id: str, output_mp3_path: Path, single_audio_file: Path = None) -> bool:
        """
        Encodes the complete recorded audio into an MP3 file.
        """
        try:
            output_mp3_path.parent.mkdir(parents=True, exist_ok=True)

            if single_audio_file and single_audio_file.exists():
                cmd = [
                    'ffmpeg', '-y',
                    '-loglevel', 'error',
                    '-i', str(single_audio_file),
                    '-codec:a', 'libmp3lame',
                    '-b:a', '192k',
                    str(output_mp3_path)
                ]
                subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if output_mp3_path.exists() and output_mp3_path.stat().st_size > 0:
                    return True

            chunk_dir = cls.get_chunk_dir(session_id)
            chunks = sorted(list(chunk_dir.glob("chunk_*.*")))
            if not chunks:
                return False

            converted_wavs = []
            for i, chunk in enumerate(chunks):
                if chunk.suffix.endswith('.whisper.wav'):
                    continue
                wav_file = chunk_dir / f"clean_{i:05d}.wav"
                cmd = [
                    'ffmpeg', '-y',
                    '-loglevel', 'error',
                    '-i', str(chunk),
                    '-ar', '44100',
                    '-ac', '2',
                    str(wav_file)
                ]
                subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if wav_file.exists():
                    converted_wavs.append(wav_file)

            if not converted_wavs:
                return False

            wav_list_file = chunk_dir / "concat_list.txt"
            with open(wav_list_file, "w") as f:
                for w in converted_wavs:
                    f.write(f"file '{w.resolve()}'\n")

            cmd = [
                'ffmpeg', '-y',
                '-loglevel', 'error',
                '-f', 'concat',
                '-safe', '0',
                '-i', str(wav_list_file),
                '-codec:a', 'libmp3lame',
                '-b:a', '192k',
                str(output_mp3_path)
            ]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            shutil.rmtree(chunk_dir, ignore_errors=True)
            return output_mp3_path.exists()
        except Exception as e:
            print(f"[AudioService] Error finalizing MP3: {e}")
            return False
