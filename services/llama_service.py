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
        groq_key = os.environ.get('GROQ_API_KEY', '').strip()
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
                with urllib.request.urlopen(req, timeout=30) as resp:
                    if resp.status == 200:
                        data = json.loads(resp.read().decode('utf-8'))
                        summary = data['choices'][0]['message']['content'].strip()
                        if summary:
                            return summary
            except Exception as e:
                print(f"[LlamaService] Groq API call error: {e}")

        # 2. Try Ollama (Local llama3:8b or llama3.2)
        ollama_url = os.environ.get('LLAMA_API_BASE', 'http://localhost:11434/api').rstrip('/')
        ollama_model = os.environ.get('LLAMA_MODEL', 'llama3:8b')
        try:
            payload = json.dumps({
                "model": ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3}
            }).encode('utf-8')
            req = urllib.request.Request(
                f"{ollama_url}/generate",
                data=payload,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode('utf-8'))
                    summary = data.get('response', '').strip()
                    if summary:
                        return summary
        except Exception:
            # Ollama not running or unreachable
            pass

        # 3. Built-in Structured Extractive Fallback Engine
        return cls._generate_fallback_summary(meeting_title, transcript_text)

    @classmethod
    def _generate_fallback_summary(cls, meeting_title: str, transcript: str) -> str:
        """
        Intelligent fallback meeting note summarizer to ensure notes and PDF generation
        work out of the box even before configuring an external Ollama / Llama 3 API endpoint.
        """
        sentences = [s.strip() for s in re.split(r'[.!?]+', transcript) if len(s.strip()) > 8]
        
        overview = sentences[:2] if sentences else ["Meeting was recorded and transcribed successfully."]
        discussion = sentences[2:7] if len(sentences) > 2 else sentences
        
        # Look for action-like words or decisions
        actions = []
        decisions = []
        for s in sentences:
            s_lower = s.lower()
            if any(w in s_lower for w in ["will", "need to", "action", "task", "should", "follow up", "assign"]):
                actions.append(s)
            if any(w in s_lower for w in ["decided", "agreed", "resolved", "plan to", "approved"]):
                decisions.append(s)

        if not actions:
            actions = ["Review meeting notes and follow up on discussed objectives."]
        if not decisions:
            decisions = ["Proceeded with the outlined agenda items."]

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
