# MIND EASE — Mental Wellness by Voice

> **Private · On-Device · Real-Time · Sub-600ms**

An on-device voice AI for mental wellness support. No cloud, no data leaving your device. Ever.

## Architecture

```
Microphone → VAD → Whisper STT → [Emotion | Mood] → Phi-3 → Kokoro TTS → Speaker
```

All 5 layers run locally via llama.cpp, whisper.cpp, and ONNX Runtime.

## Quick Start

```bash
bash setup.sh
# Then in two terminals:
cd backend && source .venv/bin/activate && python main.py
cd frontend && npm run dev
```

Open **http://localhost:3000**, tap the orb, and speak.

## Folder Structure

| Path | Contents |
|---|---|
| `engine/` | C++ inference orchestrator (whisper + llama + ONNX) |
| `backend/` | FastAPI WebSocket server + TTS |
| `frontend/` | Next.js UI with Web Audio API |
| `models/` | Downloaded model weights (gitignored) |
| `config/` | Pipeline tuning + safety responses |
| `scripts/` | Build, download, benchmark scripts |

## Latency Budget

| Layer | Target | Notes |
|---|---|---|
| VAD | 10ms | Energy gate |
| STT | 180ms | Whisper small Q4 greedy |
| Emotion + Mood | 80ms | Parallel via std::async |
| SLM | 250ms | Phi-3 Q4, 80 token cap |
| TTS | 80ms | Kokoro ONNX first chunk |
| **Total** | **≤600ms** | |

## Safety

The system monitors emotional distress across a rolling 3-turn window.
When sustained high-risk state is detected, the SLM is **bypassed** and a
predefined response + crisis resources are shown. No AI-generated content
on safety-critical paths.

**MIND EASE is not a substitute for professional mental healthcare.**
