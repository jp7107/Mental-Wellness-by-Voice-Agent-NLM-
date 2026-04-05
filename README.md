<div align="center">

# 🧠 MIND EASE

### Mental Wellness by Voice

**A fully offline, privacy-first voice AI that listens to how you feel — and talks back with empathy.**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Next.js](https://img.shields.io/badge/Next.js-16-000000?style=for-the-badge&logo=next.js&logoColor=white)](https://nextjs.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Whisper](https://img.shields.io/badge/Whisper-STT-FF6F00?style=for-the-badge&logo=openai&logoColor=white)](https://github.com/openai/whisper)
[![Phi--3](https://img.shields.io/badge/Phi--3-LLM-5C2D91?style=for-the-badge&logo=microsoft&logoColor=white)](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br/>

*No cloud. No APIs. No data leaves your device. Ever.*

---

**[Features](#-features) · [How It Works](#-how-it-works) · [Architecture](#-system-architecture) · [Quick Start](#-quick-start) · [Tech Stack](#-tech-stack)**

</div>

<br/>

## 🌟 The Problem

Over **280 million people** worldwide suffer from depression, yet most hesitate to seek help due to stigma, cost, or accessibility. Traditional therapy apps rely on cloud APIs — raising privacy concerns with the most sensitive data: *how someone feels*.

## 💡 The Solution

**MIND EASE** is a voice-first mental wellness companion that runs **entirely on your laptop**. Speak naturally, and it listens — not just to your words, but to the *emotion behind them*. It responds with genuine empathy, tracks your emotional trajectory, and automatically escalates to crisis resources when it senses you need real help.

> 🔐 **Zero data transmission.** Your voice is processed, understood, and responded to without a single byte leaving your machine.

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🎙️ Voice-First Interaction
Speak naturally — no typing, no buttons. The AI detects when you start and stop speaking automatically.

### 🧠 Emotion-Aware AI
Detects **6 emotional states** in real-time (calm, sad, anxious, angry, fearful, distressed) and tailors every response to match.

### 🔒 100% Offline & Private
Every model runs locally. Whisper for speech. Phi-3 for thinking. macOS for speaking. No internet required.

</td>
<td width="50%">

### ⚡ Real-Time Pipeline
End-to-end voice → understanding → response in **under 3 seconds**. No cloud round-trips, no loading spinners.

### 🛡️ Built-in Safety Net
A rolling mood tracker monitors distress levels. When risk is high, the AI is bypassed entirely and crisis helplines are surfaced.

### 🎨 Beautiful UI
Animated voice orb, real-time emotion display, mood visualization, and a conversation timeline — all in a sleek dark interface.

</td>
</tr>
</table>

---

## 🎯 How It Works

### Voice Interaction States

The interface communicates clearly through three distinct states:

```
  ◉ LISTENING          ◉ THINKING           ◉ RESPONDING
  ╭─────────────╮      ╭─────────────╮      ╭─────────────╮
  │             │      │    ·····    │      │   )))))))   │
  │  ((( ◯ )))  │ ──→  │    ◯       │ ──→  │      ◯     │
  │             │      │    ·····    │      │   )))))))   │
  ╰─────────────╯      ╰─────────────╯      ╰─────────────╯
   Pulsing orb          Processing dot        Audio waves
   "Speak now..."       "Understanding..."    AI voice plays
```

### User Flow

```
  ┌──────────┐     ┌──────────────┐     ┌────────────────┐     ┌──────────────┐
  │  🎙️ You  │ ──→ │ 🧠 AI Thinks │ ──→ │ 💬 AI Responds │ ──→ │ 🔊 AI Speaks │
  │  speak   │     │  + feels     │     │  with empathy  │     │  out loud    │
  └──────────┘     └──────────────┘     └────────────────┘     └──────────────┘
        │                                                             │
        └─────────────────── loop ────────────────────────────────────┘
```

**Example conversation:**

| You say | AI detects | AI responds |
|---------|-----------|-------------|
| *"I've been feeling really alone lately"* | 😢 Sad (60%) | *"Feeling alone can be incredibly heavy. You're not carrying this by yourself — I'm right here with you."* |
| *"I don't know what to do anymore"* | 😰 Anxious (65%) | *"That uncertainty is really tough. Let's take this one small step at a time together."* |
| *"Everything feels hopeless"* | 🚨 Distressed (85%) | ⚠️ **Safety activated** — Crisis resources displayed, AI bypassed |

---

## 🏗️ System Architecture

```
                              MIND EASE — Real-Time Voice AI Pipeline
 ┌─────────────────────────────────────────────────────────────────────────────────┐
 │                                                                                 │
 │   FRONTEND (Next.js + React 19)                                                │
 │   ┌─────────────┐        ┌──────────────┐        ┌─────────────┐               │
 │   │ 🎙️ Mic      │───────→│ AudioWorklet │───────→│  WebSocket  │─── PCM16 ──┐  │
 │   │ Capture     │        │ (16kHz PCM)  │        │  Client     │            │  │
 │   └─────────────┘        └──────────────┘        └─────────────┘            │  │
 │                                                                              │  │
 │   ┌─────────────┐        ┌──────────────┐        ┌─────────────┐            │  │
 │   │ 🔊 Speaker  │◀───────│ AudioContext  │◀───────│  WebSocket  │◀── b64 ──┐│  │
 │   │ Playback    │        │ (Scheduler)  │        │  Client     │          ││  │
 │   └─────────────┘        └──────────────┘        └─────────────┘          ││  │
 └──────────────────────────────────────────────────────────────────────────┬─┘│  │
                                                                           │  │  │
                                    WebSocket (ws://localhost:8000)  ◀──────┘  │  │
                                                                              │  │
 ┌────────────────────────────────────────────────────────────────────────────┼──┘
 │                                                                           │
 │   BACKEND (FastAPI + Python-Native Inference)                             │
 │                                                                           │
 │   ┌─────────┐   ┌───────────┐   ┌──────────┐   ┌────────┐   ┌────────┐  │
 │   │  1.VAD  │──→│ 2.Whisper │──→│3.Emotion │──→│ 4.LLM  │──→│ 5.TTS  │  │
 │   │ Energy  │   │faster-wsp │   │ + Mood   │   │  Phi-3 │   │ macOS  │  │
 │   │  Gate   │   │  (local)  │   │ Tracker  │   │ (GGUF) │   │  say   │  │
 │   └─────────┘   └───────────┘   └────┬─────┘   └────────┘   └────────┘  │
 │                                      │                                    │
 │                              ┌───────▼────────┐                           │
 │                              │ 🛡️ Safety      │                          │
 │                              │   Monitor      │                           │
 │                              │ (LLM Bypass)   │                           │
 │                              └────────────────┘                           │
 └───────────────────────────────────────────────────────────────────────────┘
```

### Pipeline Breakdown

| # | Layer | Technology | What It Does | Latency |
|---|-------|-----------|-------------|---------|
| 1 | **VAD** | Energy-based (Python) | Detects speech boundaries; waits for 800ms silence to segment | ~10ms |
| 2 | **STT** | faster-whisper (CTranslate2) | Converts speech to text — fully offline, mental-health vocabulary biased | ~300ms |
| 3 | **Emotion** | Keyword heuristic classifier | Scans transcript for emotional indicators across 6 categories | <1ms |
| 3b | **Mood** | Rolling window tracker | Tracks last 3 emotional scores; triggers safety if sustained distress | <1ms |
| 4 | **LLM** | Phi-3 Mini 4K (Q4 GGUF) | Generates empathetic, contextual responses via llama-cpp-python | ~1–2s |
| 5 | **TTS** | macOS `say` / Kokoro | Synthesizes natural speech and streams PCM audio back to browser | ~1s |

---

## 🚀 Quick Start

### Prerequisites

| Requirement | Version |
|-------------|---------|
| macOS | Apple Silicon recommended |
| Python | 3.11+ |
| Node.js | 18+ |
| Disk Space | ~3 GB (for AI model) |

### Setup (One-Time)

```bash
# 1. Clone
git clone https://github.com/jp7107/Mental-Wellness-by-Voice-Agent-NLM-.git
cd Mental-Wellness-by-Voice-Agent-NLM-

# 2. Backend
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install llama-cpp-python

# 3. Download AI Model (~2.3 GB)
mkdir -p ../models
curl -L "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf" \
  -o ../models/phi-3-mini-4k-q4.gguf

# 4. Frontend
cd ../frontend
npm install
```

### Run

```bash
# Terminal 1 — Start AI Backend
cd backend && source .venv/bin/activate && python main.py

# Terminal 2 — Start Web UI
cd frontend && npm run dev
```

### Use

Open **http://localhost:3000** → Tap the orb → Speak → Listen

---

## 📁 Project Structure

```
MIND EASE/
│
├── 🖥️  frontend/                    Next.js 16 Web Interface
│   ├── app/page.tsx                 Main UI (voice orb + conversation)
│   ├── components/VoiceOrb.tsx      Animated pulsing microphone orb
│   ├── components/MoodRing.tsx      Real-time mood visualization
│   ├── hooks/useAudioCapture.ts     Mic → PCM16 via AudioWorklet
│   └── hooks/useWebSocket.ts        WebSocket connection manager
│
├── ⚙️  backend/                     FastAPI Python Server
│   ├── main.py                      Server entry point
│   ├── services/pipeline_service.py Pipeline orchestrator
│   ├── services/stt_service.py      Whisper speech-to-text
│   ├── services/emotion_service.py  Emotion classifier + mood tracker
│   ├── services/llm_service.py      Phi-3 response generation
│   ├── services/tts_service.py      Text-to-speech engine
│   ├── services/vad_service.py      Voice activity detection
│   └── services/safety_service.py   Crisis detection + escalation
│
├── 🛡️  config/
│   ├── pipeline.yaml                Pipeline tuning parameters
│   └── safety_responses.yaml        Crisis responses + helpline numbers
│
├── 🧠  models/                      AI Weights (gitignored, ~2.3 GB)
├── 📜  scripts/                     Build + download utilities
└── 🎓  training/                    Model fine-tuning resources
```

---

## 🛡️ Safety System

MIND EASE includes a **mandatory, non-bypassable** safety layer:

```
  Normal Flow:    You speak → Emotion → LLM responds → TTS plays
  Safety Flow:    You speak → Emotion → ⚠️ HIGH RISK → LLM BYPASSED → Crisis resources shown
```

**When does safety activate?**
- Emotion classifier detects "distressed" or "fearful" states
- All 3 scores in the rolling mood window are ≥ 4.0
- Keywords like *"end it"*, *"give up"*, *"can't go on"* are detected

**What happens?**
- The AI model is **completely bypassed** (no generated text on critical paths)
- A pre-written, clinically appropriate response is shown
- Crisis helplines are displayed (US, India, UK)

> ⚠️ **MIND EASE is not a substitute for professional mental healthcare.** If you or someone you know is in crisis, please contact a professional immediately.

---

## 🔧 Tech Stack

<table>
<tr><th>Layer</th><th>Technology</th><th>Role</th></tr>
<tr><td>🖥️ Frontend</td><td>Next.js 16 · React 19 · Web Audio API</td><td>Voice capture, playback, real-time UI</td></tr>
<tr><td>🔌 Transport</td><td>WebSocket (FastAPI)</td><td>Bidirectional real-time audio streaming</td></tr>
<tr><td>🎙️ VAD</td><td>Custom energy-based (Python)</td><td>Speech boundary detection</td></tr>
<tr><td>📝 STT</td><td>faster-whisper (CTranslate2)</td><td>Offline speech-to-text</td></tr>
<tr><td>💭 Emotion</td><td>Keyword heuristic classifier</td><td>6-class emotion detection</td></tr>
<tr><td>🧠 LLM</td><td>Phi-3 Mini 4K (Q4) · llama-cpp-python</td><td>Empathetic response generation</td></tr>
<tr><td>🔊 TTS</td><td>macOS <code>say</code> / Kokoro</td><td>Natural speech synthesis</td></tr>
<tr><td>🛡️ Safety</td><td>YAML rule engine + mood tracker</td><td>Crisis detection & escalation</td></tr>
</table>

---

<div align="center">

### Built with ❤️ for mental wellness, privacy, and accessibility

*Because everyone deserves to be heard — even when no one else is listening.*

</div>
