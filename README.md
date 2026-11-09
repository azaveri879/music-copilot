# 🎛️ Music Copilot

> An AI-powered production assistant for musicians and producers — think Cursor for Logic Pro.

Music Copilot is a companion app that sits alongside your DAW and helps you bridge the gap between the sound in your head and the production decisions that get you there. Upload a recording or a section of your session, describe what's wrong or what vibe you're going for, and get specific, actionable guidance — not generic tutorials.

---

## Table of Contents

- [The Problem](#the-problem)
- [The Vision](#the-vision)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Backend Setup](#backend-setup)
  - [Frontend Setup](#frontend-setup)
  - [Environment Variables](#environment-variables)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Roadmap](#roadmap)
- [Contributing](#contributing)

---

## The Problem

Most AI music tools are either fully generative (Suno, Udio) or narrowly focused (auto-mastering, stem separation). Nothing exists for the **musician-first producer** — someone who records real performances, has a clear creative vision, but struggles with the technical production layer.

The gap looks like this:

- You record a guitar part that feels slightly off beat but you don't know how to fix it in Logic
- You want your track to feel "warm and lo-fi" but you don't know which plugins to reach for or what settings to use
- You hear a reference track and want to understand what production decisions make it sound the way it does
- You're mixing vocals and something feels off but you can't name what

DAWs like Logic Pro are incredibly powerful but have steep learning curves. The documentation tells you *what* a tool does, not *when* or *why* to use it in context. Music Copilot fills that gap.

---

## The Vision

**Logic Pro is the IDE. Music Copilot is the Cursor.**

Just as Cursor transformed coding by making AI context-aware of your actual codebase, Music Copilot makes AI context-aware of your actual session. The parallel is tight:

| Coding / Cursor | Music / Music Copilot |
|---|---|
| Code file | Audio session / stems |
| Linter errors | Timing issues, clipping, off-key notes |
| "Refactor this function" | "Fix the timing on bars 3–5" |
| "Make this more readable" | "Make this feel warmer / more lo-fi" |
| Cursor inline edits | Suggested Logic edits with exact steps |
| Context-aware of the codebase | Context-aware of the session |

The goal isn't to replace the producer's creative judgment — it's to collapse the gap between creative vision and technical execution.

---

## Features

### v1 (Current)

- **Audio Upload & Waveform Visualization** — Upload a full mix or a bounced stem. Scrub through it and select specific regions for analysis.
- **Region Selection** — Click and drag on the waveform to highlight a specific section ("just this chorus", "bars 8–12"). The copilot responds with advice scoped to that region.
- **Natural Language Input** — Describe what's wrong or what you want in plain English. "This feels off beat", "I want this to sound like it's underwater", "the vocals are too harsh."
- **AI-Powered Analysis** — The backend analyzes your audio for tempo, timing, key, frequency content, loudness and dynamics, then feeds that structured data to the AI layer. Selecting a region re-runs the analysis for just that section.
- **Specific, Actionable Advice** — Responses include the specific Logic Pro steps, plugin recommendations, and parameter ranges to get the result you're after. Not "try adding some reverb" — "add Space Designer, set the room size to ~40%, roll off the highs above 8kHz."
- **Session Context** — The panel remembers what you've talked about in the session, so follow-up questions work naturally ("now do the same for the chorus").
- **Streaming Responses** — Answers stream in token by token so the interaction feels fast and conversational.

### Planned (v2+)

- Reference track analysis ("make mine sound like this")
- Stem-aware analysis (upload individual tracks, not just the mix)
- Suggested settings exportable as Logic plugin presets
- macOS companion app with native DAW integration via OSC/MIDI FX
- Direct region handoff via keyboard shortcut (highlight in Logic → auto-appears in copilot)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend (React/TS)                   │
│                                                              │
│  ┌──────────────────┐    ┌──────────────────────────────┐   │
│  │  WaveformPlayer  │    │         ChatPanel            │   │
│  │  (Wavesurfer.js) │    │  - Message thread            │   │
│  │  - Upload audio  │    │  - Streaming responses       │   │
│  │  - Region select │───▶│  - Session context           │   │
│  │  - Playback      │    │  - Natural language input    │   │
│  └──────────────────┘    └──────────────────────────────┘   │
│           │                           │                      │
└───────────┼───────────────────────────┼──────────────────────┘
            │                           │
            ▼                           ▼
┌─────────────────────────────────────────────────────────────┐
│                      Backend (FastAPI)                       │
│                                                              │
│  POST /analyze              POST /chat                       │
│  ┌──────────────────┐      ┌──────────────────────────┐     │
│  │  Audio Analysis  │      │     AI Service           │     │
│  │  - librosa       │─────▶│     - Claude API         │     │
│  │  - Timing/BPM    │      │     - Audio features     │     │
│  │  - Pitch/key     │      │       as context         │     │
│  │  - Frequency     │      │     - Streaming SSE      │     │
│  │  - Dynamics      │      │     - Session memory     │     │
│  └──────────────────┘      └──────────────────────────┘     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. User uploads audio file (full mix or specific section)
2. Frontend sends file to `POST /analyze` → backend extracts audio features
3. Structured features (BPM, key, spectral data, timing issues, etc.) are stored in session state
4. Selecting a region sends the file to `POST /analyze` again with `start_time` / `end_time`. Tempo and the beat grid still come from the whole track; everything else describes the region
5. User types a natural language query in the chat panel
6. Frontend sends `POST /chat` with the query + session context (prior messages + audio features)
7. Backend constructs a rich prompt with audio analysis data and streams Claude's response back via SSE
8. Frontend renders the streaming response in the chat panel

---

## Tech Stack

### Frontend
- **React 18** + **TypeScript** — Component framework
- **Wavesurfer.js** — Waveform rendering, playback, and region selection
- **Tailwind CSS** — Styling
- **Vite** — Build tooling and dev server

### Backend
- **FastAPI** — Async Python API framework
- **librosa** — Audio analysis (BPM detection, pitch, spectral features, onset detection)
- **pyloudnorm** — ITU-R BS.1770 integrated loudness (LUFS)
- **numpy / scipy** — Signal processing
- **Anthropic Python SDK** — Claude API with streaming
- **uvicorn** — ASGI server

### AI
- **Claude** (via Anthropic API) — The reasoning layer that translates audio features + natural language into production advice

---

## Project Structure

```
music-copilot/
├── README.md
├── .gitignore
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── WaveformPlayer.tsx      # Audio upload, waveform display, region selection
│   │   │   ├── ChatPanel.tsx           # Chat UI with streaming message rendering
│   │   │   ├── MessageBubble.tsx       # Individual message component
│   │   │   └── RegionInfo.tsx          # Displays currently selected region timestamps
│   │   ├── hooks/
│   │   │   ├── useAudioAnalysis.ts     # Triggers analysis, stores audio features
│   │   │   └── useChat.ts             # Chat state, streaming, session context
│   │   ├── types/
│   │   │   └── index.ts               # Shared TypeScript types
│   │   ├── utils/
│   │   │   ├── api.ts                 # API client functions
│   │   │   └── format.ts              # Time formatting
│   │   ├── App.tsx                    # Root component, layout
│   │   └── main.tsx                   # Entry point
│   ├── index.html
│   ├── package.json
│   ├── postcss.config.js
│   ├── tsconfig.json
│   ├── vite.config.ts
│   └── tailwind.config.js
│
└── backend/
    ├── main.py                        # FastAPI app entry point, CORS, router registration
    ├── routes/
    │   ├── analyze.py                 # POST /analyze and /preview — feature extraction, AIFF playback
    │   └── chat.py                    # POST /chat — AI response with streaming SSE
    ├── services/
    │   ├── audio_analysis.py          # librosa-based audio feature extraction
    │   ├── audio_editing.py           # Gain, EQ, compression, limiting and note moves
    │   ├── audio_store.py             # Recently analyzed audio, held in memory for measuring edits
    │   └── ai_service.py             # Claude API client, prompt construction, streaming
    ├── models/
    │   └── schemas.py                 # Pydantic request/response models
    ├── tests/                         # pytest suite (synthetic audio, stubbed Claude API)
    ├── requirements.txt
    └── requirements-dev.txt
```

---

## Getting Started

### Prerequisites

- **Node.js** v18+ and **npm** or **pnpm**
- **Python** 3.12+
- An **Anthropic API key** — get one at [console.anthropic.com](https://console.anthropic.com)

---

### Backend Setup

```bash
# Navigate to the backend directory
cd music-copilot/backend

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate       # macOS/Linux
# venv\Scripts\activate        # Windows

# Install dependencies
pip install -r requirements.txt

# Add your API key (see Environment Variables)
cp .env.example .env

# Start the development server
uvicorn main:app --reload --port 8000
```

The backend will be running at `http://localhost:8000`. You can explore the auto-generated API docs at `http://localhost:8000/docs`.

---

### Frontend Setup

```bash
# Navigate to the frontend directory
cd music-copilot/frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```

The frontend will be running at `http://localhost:5173`.

---

### Environment Variables

#### Backend — create `backend/.env`

```env
# Required
ANTHROPIC_API_KEY=sk-ant-...

# Optional
CORS_ORIGINS=http://localhost:5173
MAX_FILE_SIZE_MB=50
```

#### Frontend — create `frontend/.env` (optional)

```env
VITE_API_BASE_URL=http://localhost:8000
```

`MAX_FILE_SIZE_MB` caps uploads. At the default of 50, a 24-bit stereo WAV tops out around three minutes, so raise it for full-length bounces.

---

### Running the Tests

```bash
cd backend
pip install -r requirements-dev.txt
pytest
```

The suite needs no API key. It generates its own audio and runs the Anthropic SDK against a local stand-in for the API.

---

## Usage

### Basic workflow

1. **Upload audio** — Click the upload area or drag and drop a `.wav`, `.mp3`, `.aiff`, or `.flac` file. The waveform renders automatically.

2. **Select a region (optional)** — Click and drag on the waveform to select a specific section you want to focus on. The timestamps appear above the chat input. If no region is selected, the copilot analyzes the full file.

3. **Ask your question** — Type naturally in the chat panel:
   - *"This guitar part feels off beat, can you help me fix it?"*
   - *"I want this to feel like a late-night drive — dark, reverb-y, slightly washed out"*
   - *"The low end is muddy. What should I cut?"*
   - *"What key is this in and does the melody clash with it anywhere?"*

4. **Get specific advice** — The copilot analyzes your audio, understands the context of your question, and responds with actionable steps — specific to Logic Pro, specific to your file.

5. **Try an edit** — Ask for a change, like *"make this warmer"*. The copilot can propose specific edits, which you review and apply to a copy of your file. Switch between Original and Edited to compare, then download the WAV and bring it into Logic. Your original file is never changed. Tone edits are on by default; note moves start switched off, so you approve each one.

6. **Adjust the layout** — Drag the divider beside the chat panel to widen or narrow it, or focus it and use the arrow keys. Double-click to reset. Your width is remembered.

7. **Follow up** — The session remembers everything. Ask follow-up questions naturally: *"Now do the same thing for bars 24–32"* or *"What if I wanted a warmer sound instead?"*

---

## API Reference

### `POST /analyze`

Accepts an audio file upload. Returns structured audio features.

**Request:** `multipart/form-data`
- `file`: audio file (`.wav`, `.mp3`, `.aiff`, `.aif`, `.flac`)
- `start_time` *(optional)*: float, start of region in seconds
- `end_time` *(optional)*: float, end of region in seconds

`timing_issues` are candidates to check by ear. They're measured against a grid at the single tempo that best fits the track. If no steady tempo fits, `timing_checked` is `false` and no issues are reported, rather than guessing against an unreliable grid. Notes near the midpoint between grid lines and unusually quiet attacks are left out, and an exact digital loop reports each spot once.

With a region, `bpm` and the beat grid come from the whole track. `timing_issues` are limited to the region, with timestamps as positions in the file. All other fields describe the region. Fields that can't be measured, such as loudness on a clip under 400ms or anything on silence, are `null`.

`bpm_confident` and `key_confident` are `false` when a reading is only a guess, which is common on sparse parts. `bpm` and `key` still hold that guess, but the app doesn't show it or pass it to Claude. Key and `dynamic_range_db` ignore moments more than 40 dB below the loudest point, so gaps in an exported track don't skew them.

**Errors:** `400` for an unsupported type, undecodable audio or an invalid region; `413` for a file over `MAX_FILE_SIZE_MB`.

**Response:**
```json
{
  "session_id": "uuid",
  "duration": 42.3,
  "sample_rate": 44100,
  "bpm": 92.4,
  "bpm_confident": true,
  "key": "C minor",
  "key_confident": true,
  "loudness_lufs": -14.2,
  "peak_dbfs": -0.8,
  "dynamic_range_db": 8.1,
  "spectral_centroid_mean": 2340.5,
  "onset_strength_mean": 0.43,
  "timing_checked": true,
  "timing_issues": [
    { "timestamp": 3.21, "description": "Onset ~38ms late against the beat grid" }
  ],
  "frequency_profile": {
    "low_energy": 0.42,
    "mid_energy": 0.35,
    "high_energy": 0.23
  },
  "band_levels": [
    { "name": "sub", "low_hz": 20, "high_hz": 60, "level_db": -31.4 }
  ],
  "prominent_frequencies_hz": [158, 188, 305, 621],
  "peak_moments": [
    { "timestamp": 125.31, "level_dbfs": -0.1 }
  ]
}
```

`band_levels` covers six bands from 20 Hz to 20 kHz, in dB relative to the loudest band. `prominent_frequencies_hz` lists where energy stands out between 80 Hz and 1 kHz, which may just be the notes played. `peak_moments` lists up to five moments within 6 dB of the loudest one, as positions in the file. These all describe the selected region when there is one.

---

### `POST /preview`

Accepts the same `file` field as `/analyze` and returns a 16-bit WAV copy (`audio/wav`). The frontend uses it to draw and play AIFF files, which Chrome and Firefox can't decode.

---

### `POST /edit`

Applies edits to a copy of the audio and returns a 24-bit WAV at the original sample rate and channel count. A JSON summary is in the `X-Edit-Report` header.

**Request:** `multipart/form-data`
- `file`: audio file
- `edits`: JSON array, applied with note moves first
- `start_time`, `end_time` *(optional)*: limits tone edits to this region. Note moves use their own position

```json
[
  { "type": "gain", "gain_db": -2 },
  { "type": "eq", "eq_type": "peak", "freq_hz": 300, "gain_db": -3, "q": 1.2 },
  { "type": "compressor", "threshold_db": -18, "ratio": 3, "attack_ms": 10, "release_ms": 120, "makeup_db": 2 },
  { "type": "limiter", "ceiling_db": -1, "input_gain_db": 3, "release_ms": 100 },
  { "type": "nudge", "onset_time": 1.74, "shift_ms": -39 }
]
```

A limiter always runs after other tone edits. `eq_type` is `peak`, `low_shelf`, `high_shelf`, `highpass` or `lowpass`. In an exact digital loop, a `nudge` moves that note in every repeat. If the result would clip, the whole file is turned down to -0.3 dBFS and the report says by how much.

Edits use the app's own processing rather than Logic's plugins, so similar settings won't sound identical. Note moves work best on solo tracks, because moving a note in a mix moves everything playing at that moment.

---

### `POST /edit/estimate`

Takes the same fields as `/edit` and returns what the edits would do to loudness and peak level, without keeping the result. With a region, the figures cover that region.

```json
{
  "before_lufs": -17.2, "after_lufs": -13.9,
  "before_peak_dbfs": -0.1, "after_peak_dbfs": -1.0,
  "turned_down_db": 0.0, "nudges_applied": 0
}
```

---

### `POST /chat`

Sends a user message and streams back an AI response using Server-Sent Events (SSE).

**Request:** `application/json`
```json
{
  "session_id": "uuid",
  "message": "The guitar feels a bit off beat in the first section",
  "audio_features": { ... },
  "region": { "start": 0.0, "end": 8.5 },
  "history": [
    { "role": "user", "content": "..." },
    { "role": "assistant", "content": "..." }
  ],
  "file_type": "track",
  "song_bpm": 98
}
```

`file_type` (`track`, `mix` or `master`) and `song_bpm` (20–300) are optional. They're what the user told the app about the audio, and `song_bpm` replaces the detected tempo.

**Response:** `text/event-stream`
```
data: {"type": "delta", "content": "Looking at your audio"}
data: {"type": "delta", "content": ", I can see a few timing issues..."}
data: {"type": "done"}
```

When the copilot proposes edits, the backend measures them on the audio from the matching `/analyze` call and sends the result back to Claude, which can revise up to twice. The final proposal then arrives as `data: {"type": "proposal", "proposal": {"summary": "...", "edits": [...]}}` before `done`. Nothing is applied until the client calls `/edit`. The backend holds the two most recently analyzed files in memory for this.

Failures after the stream has started arrive as `data: {"type": "error", "content": "..."}`.

---

## Roadmap

### v1 — Core Loop (current)
- [x] Project scaffold and architecture
- [x] Audio upload and waveform rendering
- [x] Region selection UI
- [x] Backend audio analysis (librosa)
- [x] Chat panel with streaming responses
- [x] Session context (conversation memory within a session)

### v2 — Deeper Analysis
- [ ] Stem-aware analysis (upload individual tracks)
- [ ] Reference track upload ("make mine sound like this")
- [ ] Pitch correction guidance (detect off-key notes with timestamps)
- [ ] Clip gain and automation suggestions

### v3 — DAW Integration
- [ ] macOS native companion app (SwiftUI or Electron)
- [ ] Logic Pro region handoff via OSC — select a region in Logic, it appears in the copilot automatically
- [ ] Export suggested settings as Logic plugin presets (.pst files)
- [ ] Keyboard shortcut to send selected region to copilot without manual export

---

## Contributing

This is an early-stage project. If you're a producer or engineer who wants to shape what this becomes, open an issue or reach out.

```bash
# Clone the repo
git clone https://github.com/azaveri879/music-copilot.git
cd music-copilot

# Set up both environments (see Getting Started above)
# Make your changes on a feature branch
git checkout -b feature/your-feature-name

# Submit a PR with a clear description of what it does and why
```

---

*Built by a musician who got tired of not knowing what to do with his guitar recordings.*
