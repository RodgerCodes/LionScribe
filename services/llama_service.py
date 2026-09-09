import os
import re
import json
import urllib.request
import urllib.error
from typing import Dict, Any

class LlamaService:
    @classmethod
    def generate_meeting_summary(cls, meeting_title: str, transcript_text: str) -> str:
        """
        Uses Llama 3 model (not the highest tier: e.g. llama3:8b / llama3.2 / llama-3.1-8b-instant)
        to summarize transcription into a structured notes format.
        """
        if not transcript_text or not transcript_text.strip():
            return (
                "### 📌 Executive Summary\n"
                "_No transcription content was recorded for this meeting._"
            )

        prompt = (
            f"You are an executive assistant taking structured meeting notes. "
            f"Below is the raw transcription of a meeting titled '{meeting_title}'.\n\n"
            f"Please generate a comprehensive, highly organized summary in markdown notes format with the following exact sections:\n\n"
            f"### 📌 Executive Summary\n"
            f"A concise 2-3 sentence overview of the meeting's main purpose and outcome.\n\n"
            f"### 🔑 Key Discussion Points\n"
            f"Bullet points capturing the core topics discussed in detail.\n\n"
            f"### 🎯 Decisions Made\n"
            f"Explicit decisions and agreements established during the discussion.\n\n"
            f"### 📋 Action Items & Next Steps\n"
            f"Clear checklist of tasks, action items, and who is responsible.\n\n"
            f"Meeting Transcript:\n\"\"\"\n{transcript_text}\n\"\"\"\n\n"
            f"Respond directly with the markdown formatted notes only."
        )

        # 1. Try Groq (Llama-3.1-8B-Instant) if GROQ_API_KEY is present
        from config import Config
        groq_key = os.environ.get('GROQ_API_KEY', '').strip() or getattr(Config, 'GROQ_API_KEY', '').strip()
        if groq_key:
            try:
                payload = json.dumps({
                    "model": "llama-3.1-8b-instant",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3
                }).encode('utf-8')
                req = urllib.request.Request(
                    "https://api.groq.com/openai/v1/chat/completions",
                    data=payload,
                    headers={
                        "Authorization": f"Bearer {groq_key}",
                        "Content-Type": "application/json"
                    }
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    if resp.status == 200:
                        data = json.loads(resp.read().decode('utf-8'))
                        summary = data['choices'][0]['message']['content'].strip()
                        if summary:
                            return summary
            except Exception as e:
                print(f"[LlamaService] Groq API notice: {e}")

        # 2. Try OpenAI-compatible endpoint (OpenRouter, local vLLM, LM Studio)
        llama_key = os.environ.get('LLAMA_API_KEY', '').strip() or getattr(Config, 'LLAMA_API_KEY', '').strip()
        llama_base = os.environ.get('LLAMA_API_BASE', '').strip() or getattr(Config, 'LLAMA_API_BASE', '').strip()
        llama_model = os.environ.get('LLAMA_MODEL', 'llama3:8b').strip() or getattr(Config, 'LLAMA_MODEL', 'llama3:8b')

        if llama_key and ('/v1' in llama_base or 'openrouter' in llama_base.lower()):
            try:
                endpoint = f"{llama_base.rstrip('/')}/chat/completions"
                payload = json.dumps({
                    "model": llama_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3
                }).encode('utf-8')
                req = urllib.request.Request(
                    endpoint,
                    data=payload,
                    headers={
                        "Authorization": f"Bearer {llama_key}",
                        "Content-Type": "application/json"
                    }
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    if resp.status == 200:
                        data = json.loads(resp.read().decode('utf-8'))
                        summary = data['choices'][0]['message']['content'].strip()
                        if summary:
                            return summary
            except Exception as e:
                print(f"[LlamaService] Custom API notice: {e}")

        # 3. Try Local Ollama (short 5s timeout to prevent UI hanging)
        ollama_url = (llama_base or 'http://localhost:11434/api').rstrip('/')
        if 'localhost' in ollama_url or '127.0.0.1' in ollama_url:
            try:
                payload = json.dumps({
                    "model": llama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3}
                }).encode('utf-8')
                req = urllib.request.Request(
                    f"{ollama_url}/generate",
                    data=payload,
                    headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=5) as resp:
                    if resp.status == 200:
                        data = json.loads(resp.read().decode('utf-8'))
                        summary = data.get('response', '').strip()
                        if summary:
                            return summary
            except Exception as e:
                print(f"[LlamaService] Local Ollama notice: {e}")

        # 4. Built-in Structured Extractive Fallback Engine (Immediate, 0s latency)
        return cls._generate_fallback_summary(meeting_title, transcript_text)

    @classmethod
    def _generate_fallback_summary(cls, meeting_title: str, transcript: str) -> str:
        """
        Intelligent fallback meeting note summarizer to ensure notes and PDF generation
        work out of the box even before configuring an external Ollama / Llama 3 API endpoint.
        """
        # Strip timestamps e.g. [00:01] from sentences for clean notes
        clean_text = re.sub(r'\[\d{1,2}:\d{2}(?::\d{2})?\]', '', transcript).strip()
        sentences = [s.strip() for s in re.split(r'[.!?]+', clean_text) if len(s.strip()) > 8]
        
        overview = sentences[:2] if sentences else ["Meeting was recorded and transcribed successfully."]
        discussion = sentences[2:7] if len(sentences) > 2 else sentences
        
        # Look for action-like words or decisions
        actions = []
        decisions = []
        for s in sentences:
            s_lower = s.lower()
            if any(w in s_lower for w in ["will", "need to", "action", "task", "should", "follow up", "assign", "plan"]):
                actions.append(s)
            if any(w in s_lower for w in ["decided", "agreed", "resolved", "approved", "confirmed"]):
                decisions.append(s)

        if not actions:
            actions = ["Review meeting discussion points and follow up with team members."]
        if not decisions:
            decisions = ["Proceeded with discussed agenda items."]

        return (
            f"### 📌 Executive Summary\n"
            f"{' '.join(overview)}.\n\n"
            f"### 🔑 Key Discussion Points\n"
            + "\n".join([f"- {d}" for d in discussion[:6]]) + "\n\n"
            f"### 🎯 Decisions Made\n"
            + "\n".join([f"- {d}" for d in decisions[:4]]) + "\n\n"
            f"### 📋 Action Items & Next Steps\n"
            + "\n".join([f"- [ ] {a}" for a in actions[:5]])
        )
