document.addEventListener('DOMContentLoaded', () => {
    const config = window.MEETING_CONFIG;
    if (!config) return;

    // DOM Elements
    const btnStart = document.getElementById('btn-start');
    const btnEndMeeting = document.getElementById('btn-end-meeting');
    const startPanel = document.getElementById('start-panel');
    const activePanel = document.getElementById('active-panel');
    const recordingGlow = document.getElementById('recording-glow');
    const meetingBadge = document.getElementById('meeting-badge');
    const timerDisplay = document.getElementById('timer-display');
    const micStatusLabel = document.getElementById('mic-status-label');
    const transcriptionStatus = document.getElementById('transcription-status');
    const transcriptPlaceholder = document.getElementById('transcript-placeholder');
    const transcriptStream = document.getElementById('transcript-stream');
    const interimStream = document.getElementById('interim-stream');
    const transcriptBox = document.getElementById('transcript-box');
    const btnCopyTranscript = document.getElementById('btn-copy-transcript');
    const processingModal = document.getElementById('processing-modal');
    const processingStatusText = document.getElementById('processing-status-text');

    // Audio & State Variables
    let audioContext = null;
    let micStream = null;
    let processorNode = null;
    let micSource = null;
    let pcmBuffer = [];
    let allRecordedBuffers = [];
    let chunkCounter = 0;
    let chunkInterval = null;
    let timerInterval = null;
    let startTime = null;
    let fullTranscriptText = "";
    let isRecording = false;
    let speechRecognition = null;
    let ws = null;
    let lastVoiceTime = Date.now();
    let accumulatedSamples = 0;

    const TARGET_SAMPLE_RATE = 16000;
    const SILENCE_THRESHOLD = 0.015; // RMS energy threshold
    const MAX_CHUNK_DURATION_SEC = 3.5;

    // Timer functions
    function startTimer() {
        startTime = Date.now();
        timerInterval = setInterval(() => {
            const elapsedMs = Date.now() - startTime;
            const totalSecs = Math.floor(elapsedMs / 1000);
            const hrs = String(Math.floor(totalSecs / 3600)).padStart(2, '0');
            const mins = String(Math.floor((totalSecs % 3600) / 60)).padStart(2, '0');
            const secs = String(totalSecs % 60).padStart(2, '0');
            timerDisplay.textContent = `${hrs}:${mins}:${secs}`;
        }, 1000);
    }

    function stopTimer() {
        if (timerInterval) {
            clearInterval(timerInterval);
            timerInterval = null;
        }
        return startTime ? Math.floor((Date.now() - startTime) / 1000) : 0;
    }

    // Convert Float32Array PCM samples to 16-bit PCM WAV ArrayBuffer
    function encodeWAV(samples, sampleRate) {
        const buffer = new ArrayBuffer(44 + samples.length * 2);
        const view = new DataView(buffer);

        function writeString(offset, str) {
            for (let i = 0; i < str.length; i++) {
                view.setUint8(offset + i, str.charCodeAt(i));
            }
        }

        writeString(0, 'RIFF');
        view.setUint32(4, 36 + samples.length * 2, true);
        writeString(8, 'WAVE');
        writeString(12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true);
        view.setUint16(22, 1, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * 2, true);
        view.setUint16(32, 2, true);
        view.setUint16(34, 16, true);
        writeString(36, 'data');
        view.setUint32(40, samples.length * 2, true);

        let offset = 44;
        for (let i = 0; i < samples.length; i++, offset += 2) {
            const s = Math.max(-1, Math.min(1, samples[i]));
            view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
        }

        return buffer;
    }

    // Downsample buffer to 16000Hz
    function downsampleBuffer(buffer, inputRate, outputRate) {
        if (inputRate === outputRate) {
            return new Float32Array(buffer);
        }
        const sampleRateRatio = inputRate / outputRate;
        const newLength = Math.round(buffer.length / sampleRateRatio);
        const result = new Float32Array(newLength);
        let offsetResult = 0;
        let offsetBuffer = 0;
        while (offsetResult < result.length) {
            const nextOffsetBuffer = Math.round((offsetResult + 1) * sampleRateRatio);
            let accum = 0, count = 0;
            for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
                accum += buffer[i];
                count++;
            }
            result[offsetResult] = count > 0 ? accum / count : 0;
            offsetResult++;
            offsetBuffer = nextOffsetBuffer;
        }
        return result;
    }

    // Append transcript segment to UI
    function appendTranscriptSegment(text, isInterim = false) {
        if (!text || !text.trim()) return;

        if (transcriptPlaceholder) {
            transcriptPlaceholder.classList.add('hidden');
        }

        if (isInterim) {
            interimStream.textContent = text;
            return;
        }

        interimStream.textContent = "";

        const cleanText = text.trim();
        // Avoid duplicate consecutive lines
        if (fullTranscriptText.endsWith(cleanText)) return;

        const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        const segmentDiv = document.createElement('div');
        segmentDiv.className = 'p-3.5 bg-slate-900/90 rounded-xl border border-slate-800 text-slate-100 text-sm shadow-sm flex items-start gap-3 transition-all duration-300';
        segmentDiv.innerHTML = `
            <span class="text-[10px] font-mono text-emerald-400 bg-emerald-950/80 px-2 py-0.5 rounded border border-emerald-800/60 select-none mt-0.5 font-bold">${timeStr}</span>
            <span class="flex-grow leading-relaxed">${cleanText}</span>
        `;
        transcriptStream.appendChild(segmentDiv);

        fullTranscriptText += (fullTranscriptText ? " " : "") + cleanText;
        transcriptBox.scrollTop = transcriptBox.scrollHeight;
    }

    // WebSocket Initialization
    function initWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/transcribe/${config.sessionId}`;
        
        try {
            ws = new WebSocket(wsUrl);
            ws.binaryType = 'arraybuffer';

            ws.onopen = () => {
                transcriptionStatus.innerHTML = '<span class="w-2 h-2 rounded-full bg-emerald-400"></span> Live (Distil-Whisper)';
            };

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    if (data.chunk_text) {
                        appendTranscriptSegment(data.chunk_text, false);
                    }
                } catch (e) {}
            };

            ws.onerror = () => {};
        } catch (err) {}
    }

    // Smart VAD Chunk Processing: Transcribes full phrases when pauses occur
    async function processAndSendChunk(force = false) {
        if (!isRecording || pcmBuffer.length === 0) return;

        const bufferDurationSec = accumulatedSamples / (audioContext ? audioContext.sampleRate : 44100);
        const timeSinceVoice = (Date.now() - lastVoiceTime) / 1000;

        // Send if:
        // 1. Force send (end of meeting)
        // 2. We have speech and a natural pause (> 350ms silence) has occurred
        // 3. Buffer has reached maximum phrase duration (3.5s)
        const shouldSend = force || 
                           (bufferDurationSec >= 1.0 && timeSinceVoice >= 0.35) || 
                           (bufferDurationSec >= MAX_CHUNK_DURATION_SEC);

        if (!shouldSend) return;

        const currentPcm = pcmBuffer;
        pcmBuffer = [];
        accumulatedSamples = 0;

        let totalLen = 0;
        for (const b of currentPcm) totalLen += b.length;
        const merged = new Float32Array(totalLen);
        let offset = 0;
        for (const b of currentPcm) {
            merged.set(b, offset);
            offset += b.length;
        }

        const downsampled = downsampleBuffer(merged, audioContext.sampleRate, TARGET_SAMPLE_RATE);
        if (downsampled.length < TARGET_SAMPLE_RATE * 0.4) {
            return; // Skip negligible noise
        }

        const wavArrayBuffer = encodeWAV(downsampled, TARGET_SAMPLE_RATE);

        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(wavArrayBuffer);
            return;
        }

        // HTTP Fallback
        try {
            const formData = new FormData();
            const wavBlob = new Blob([wavArrayBuffer], { type: 'audio/wav' });
            formData.append('audio', wavBlob, `chunk_${chunkCounter++}.wav`);
            formData.append('chunk_index', chunkCounter);
            formData.append('mime_type', 'audio/wav');

            const res = await fetch(config.apiUrlChunk, { method: 'POST', body: formData });
            if (res.ok) {
                const data = await res.json();
                if (data.chunk_text) {
                    appendTranscriptSegment(data.chunk_text, false);
                }
            }
        } catch (err) {
            console.error('Audio chunk upload:', err);
        }
    }

    // Real-Time Browser Speech Recognition (Instant Word-by-Word Rendering)
    function initSpeechRecognition() {
        const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRec) return;

        try {
            speechRecognition = new SpeechRec();
            speechRecognition.continuous = true;
            speechRecognition.interimResults = true;
            speechRecognition.lang = 'en-US';

            speechRecognition.onresult = (event) => {
                let interim = '';
                for (let i = event.resultIndex; i < event.results.length; ++i) {
                    if (event.results[i].isFinal) {
                        const finalText = event.results[i][0].transcript.trim();
                        if (finalText) {
                            appendTranscriptSegment(finalText, false);
                        }
                    } else {
                        interim += event.results[i][0].transcript;
                    }
                }
                if (interim) {
                    appendTranscriptSegment(interim, true);
                }
            };

            speechRecognition.onend = () => {
                if (isRecording && speechRecognition) {
                    try { speechRecognition.start(); } catch (e) {}
                }
            };

            speechRecognition.start();
        } catch (e) {
            console.warn('SpeechRecognition init:', e);
        }
    }

    // Start Meeting
    async function startMeeting() {
        try {
            micStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                }
            });

            initWebSocket();

            const AudioCtx = window.AudioContext || window.webkitAudioContext;
            audioContext = new AudioCtx();
            micSource = audioContext.createMediaStreamSource(micStream);

            processorNode = audioContext.createScriptProcessor(4096, 1, 1);
            pcmBuffer = [];
            allRecordedBuffers = [];
            chunkCounter = 0;
            accumulatedSamples = 0;
            lastVoiceTime = Date.now();

            processorNode.onaudioprocess = (e) => {
                if (!isRecording) return;
                const inputData = e.inputBuffer.getChannelData(0);
                const clone = new Float32Array(inputData);
                
                // Calculate RMS energy for smart VAD
                let sumSquares = 0;
                for (let i = 0; i < inputData.length; i++) {
                    sumSquares += inputData[i] * inputData[i];
                }
                const rms = Math.sqrt(sumSquares / inputData.length);
                if (rms > SILENCE_THRESHOLD) {
                    lastVoiceTime = Date.now();
                }

                pcmBuffer.push(clone);
                allRecordedBuffers.push(clone);
                accumulatedSamples += clone.length;
            };

            micSource.connect(processorNode);
            processorNode.connect(audioContext.destination);

            isRecording = true;

            // Check VAD and dispatch phrases periodically
            chunkInterval = setInterval(() => processAndSendChunk(false), 500);

            await fetch(config.apiUrlStart, { method: 'POST' });

            // UI Transitions
            startPanel.classList.add('hidden');
            activePanel.classList.remove('hidden');
            recordingGlow.classList.remove('hidden');

            meetingBadge.className = "px-2.5 py-0.5 rounded-full text-xs font-semibold bg-rose-950 text-rose-400 border border-rose-800 animate-pulse flex items-center gap-1.5";
            meetingBadge.innerHTML = '<span class="w-2 h-2 rounded-full bg-rose-400"></span> Live Recording';

            micStatusLabel.innerHTML = '<i class="fa-solid fa-microphone text-emerald-400 animate-pulse"></i> Live Microphone Active';
            transcriptionStatus.innerHTML = '<span class="w-2 h-2 rounded-full bg-emerald-400"></span> Listening with Distil-Whisper...';

            startTimer();
            initSpeechRecognition();

        } catch (err) {
            console.error('Failed to start recording:', err);
            alert('Unable to access microphone. Please check your browser permissions.');
        }
    }

    // End Meeting & Generate Summary
    async function endMeeting() {
        if (!isRecording) return;
        isRecording = false;

        const durationSeconds = stopTimer();

        if (chunkInterval) {
            clearInterval(chunkInterval);
            chunkInterval = null;
        }

        if (speechRecognition) {
            try { speechRecognition.stop(); } catch (e) {}
            speechRecognition = null;
        }

        processingModal.classList.remove('hidden');

        if (processorNode) processorNode.disconnect();
        if (micSource) micSource.disconnect();
        if (micStream) micStream.getTracks().forEach(track => track.stop());

        // Process last phrase
        await processAndSendChunk(true);

        // Build full WAV recording blob
        let totalLen = 0;
        for (const b of allRecordedBuffers) totalLen += b.length;
        const mergedAll = new Float32Array(totalLen);
        let offset = 0;
        for (const b of allRecordedBuffers) {
            mergedAll.set(b, offset);
            offset += b.length;
        }

        const downsampledAll = downsampleBuffer(mergedAll, audioContext ? audioContext.sampleRate : 44100, TARGET_SAMPLE_RATE);
        const fullWavBuffer = encodeWAV(downsampledAll, TARGET_SAMPLE_RATE);
        const fullWavBlob = new Blob([fullWavBuffer], { type: 'audio/wav' });

        if (ws) {
            try { ws.close(); } catch (e) {}
            ws = null;
        }

        if (audioContext && audioContext.state !== 'closed') {
            audioContext.close();
        }

        // Animate modal steps
        const stepAudio = document.getElementById('step-audio');
        const stepTxt = document.getElementById('step-txt');
        const stepLlama = document.getElementById('step-llama');
        const stepPdf = document.getElementById('step-pdf');

        setTimeout(() => {
            stepAudio.className = "flex items-center gap-2.5 text-emerald-400 font-medium";
            stepAudio.innerHTML = '<i class="fa-solid fa-check text-emerald-400"></i><span>1. Audio finalized (.mp3)</span>';
            stepTxt.className = "flex items-center gap-2.5 text-emerald-400 font-medium";
            stepTxt.innerHTML = '<i class="fa-solid fa-check text-emerald-400"></i><span>2. Full transcript exported (.txt)</span>';
            stepLlama.className = "flex items-center gap-2.5 text-teal-400 font-medium";
            stepLlama.innerHTML = '<i class="fa-solid fa-circle-notch animate-spin text-teal-400"></i><span>3. Prompting Llama 3 for meeting notes...</span>';
        }, 1200);

        try {
            const formData = new FormData();
            if (fullWavBlob.size > 44) {
                formData.append('full_audio', fullWavBlob, 'recording.wav');
            }
            formData.append('duration_seconds', durationSeconds);
            formData.append('final_transcript', fullTranscriptText);

            processingStatusText.textContent = "Llama 3 is analyzing meeting notes & building PDF summary...";

            const response = await fetch(config.apiUrlEnd, {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                const result = await response.json();
                
                stepLlama.className = "flex items-center gap-2.5 text-emerald-400 font-medium";
                stepLlama.innerHTML = '<i class="fa-solid fa-check text-emerald-400"></i><span>3. Llama 3 notes created</span>';
                
                stepPdf.className = "flex items-center gap-2.5 text-emerald-400 font-medium";
                stepPdf.innerHTML = '<i class="fa-solid fa-check text-emerald-400"></i><span>4. Summary PDF generated!</span>';

                setTimeout(() => {
                    if (result.redirect_url) {
                        window.location.href = result.redirect_url;
                    }
                }, 800);
            } else {
                throw new Error('Server returned error during finalization');
            }
        } catch (err) {
            console.error('Error finalizing meeting:', err);
            window.location.href = `/meetings/${config.sessionId}/summary`;
        }
    }

    // Event Listeners
    if (btnStart) {
        btnStart.addEventListener('click', startMeeting);
    }

    if (btnEndMeeting) {
        btnEndMeeting.addEventListener('click', () => {
            if (confirm('End meeting and generate Llama 3 summary & notes PDF?')) {
                endMeeting();
            }
        });
    }

    if (btnCopyTranscript) {
        btnCopyTranscript.addEventListener('click', () => {
            if (!fullTranscriptText) {
                alert('No transcript content yet to copy.');
                return;
            }
            navigator.clipboard.writeText(fullTranscriptText).then(() => {
                const original = btnCopyTranscript.innerHTML;
                btnCopyTranscript.innerHTML = '<i class="fa-solid fa-check text-emerald-400"></i> Copied!';
                setTimeout(() => { btnCopyTranscript.innerHTML = original; }, 2000);
            });
        });
    }
});
