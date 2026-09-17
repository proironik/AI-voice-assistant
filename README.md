# Angel — Local AI Voice Assistant

A fully local, GPU-accelerated voice assistant with a browser UI. You type (or
speak, via the desktop client), an LLM replies in character, the reply is spoken
in a **cloned voice**, and natural-language commands like *"play some Drake"* or
*"open notepad"* are carried out on the machine — all running on your own
hardware, no cloud APIs.

The voice pipeline is **Ollama (LLM) → Chatterbox-Turbo (TTS) → RVC (voice
conversion)**, orchestrated by a small threaded web server that also serves the
gold-themed web UI.

> This repository contains the **desktop assistant** only. The companion phone
> app and large model weights are intentionally excluded.

---

## Features

- **Local LLM chat** — replies via [Ollama](https://ollama.com) (`mistral`), kept
  resident in VRAM and pre-warmed so the first message isn't slow.
- **Cloned-voice speech** — Chatterbox-Turbo TTS converted through RVC, so Angel
  speaks in a custom voice rather than a generic TTS voice.
- **Expression handling** — native voice cues (`[laugh]`, `[chuckle]`, `[sigh]`,
  `[gasp]`, …) are kept for speech but hidden from the visible text; unsupported
  or malformed tags are stripped so they're never read aloud.
- **Semantic commands** — natural phrasing is matched to actions using
  `all-minilm` embeddings, so *"I wanna watch some Mr Beast"* routes the same as
  *"search YouTube for Mr Beast"*. Falls back to fast literal keyword matching.
- **Music control**
  - Songs / artists / albums resolve on **YouTube Music** (exact track, artist
    shuffle, or album) via the unofficial `ytmusicapi`.
  - **Apple Music** on request — opens the installed Windows app (the app
    can't be deep-linked by URL, so Angel opens it and tells you to hit play).
- **Media transport** — pause / resume / next / previous and *"what's playing?"*
  drive whatever is playing on the machine (Spotify, YouTube Music, etc.) through
  the Windows media session, and never Angel's own voice.
- **App control (safe mode)** — *"open chrome"*, *"close spotify"*. Closing is
  graceful (windows get a close request, so unsaved-work prompts still appear)
  and a denylist protects system processes and the assistant's own stack.
- **Web UI** — circular audio visualizer, input/output boxes, live service
  status pills, a fixed media player, and a start/stop-services button. The
  background and elements react to whatever music is playing.
- **Safe service lifecycle** — start/stop all services from the UI; graceful
  shutdown so the TTS/RVC servers unload their models and release VRAM cleanly.

---

## Architecture

```
Browser UI  ──HTTP──▶  web_server.py (:8080)
                          │
        ┌─────────────────┼───────────────────────┐
        ▼                 ▼                         ▼
   Ollama (:11434)   Chatterbox TTS (:7866)    RVC (:7865)
   mistral + all-       tts_server.py           rvc_server.py
   minilm embeddings    (cloned voice)          (voice conversion)
```

| Component | File | Port | Role |
|-----------|------|------|------|
| Web server / UI | `web_server.py` | 8080 | Routes chat, speech, commands, media; serves `web/` |
| LLM chat | `convo.py` | 11434 (Ollama) | `mistral` conversation, kept warm |
| Semantic intent | `intent.py`, `slots.py` | 11434 | `all-minilm` embedding match → command |
| Commands | `commands.py` | — | Maps intent to a deferred action + spoken reply |
| Expression | `expressions.py` | — | Voice-cue whitelist, hides tags from visible text |
| TTS | `tts.py`, `tts_server.py` | 7866 | Chatterbox-Turbo synthesis |
| Voice conversion | `rvc.py`, `rvc_server.py` | 7865 | RVC clone conversion |
| Music | `ytmusic.py`, `applemusic.py` | — | Song/artist/album resolution |
| Media transport | `playback.py` | — | Windows media-session control |
| App control | `appcontrol.py` | — | Open/close desktop apps (safe mode) |
| Audio reactivity | `audiolevels.py` | — | System-audio levels for the visualizer |
| Startup | `start_all.ps1` | — | Launches all services GPU-first |

---

## Requirements

- **Windows** (uses the Windows media session and `winsound`).
- An **NVIDIA GPU** is strongly recommended — the LLM, TTS, and RVC all run on
  it. The pipeline was tuned for an 8 GB card by loading Ollama *after* the voice
  models so it isn't forced onto CPU.
- **Ollama** with the models pulled:
  ```
  ollama pull mistral
  ollama pull all-minilm
  ```
- Three separate Python environments, because the stacks pin conflicting Torch
  versions:
  - the **assistant venv** (this repo),
  - a **TTS env** for Chatterbox-Turbo (`torch==2.6.0`),
  - an **RVC env** for RVC (`torch 2.5.1+cu118`).
- Optional command dependencies (each command degrades gracefully if absent):
  ```
  .\venv\Scripts\python.exe -m pip install -r requirements-commands.txt
  ```

> Model weights, the RVC WebUI, and the per-service environments live outside
> this repo and are not included.

---

## Running it

1. **Start the services** (LLM, TTS, RVC) — from the UI's *Start All Services*
   button, or:
   ```powershell
   .\start_all.ps1
   ```
   Models take ~20–30 s to load and warm the first time.

2. **Start the web server:**
   ```powershell
   .\venv\Scripts\python.exe web_server.py
   ```
   Open **http://localhost:8080**.

3. **(Optional) reach it from another device on your LAN** — the server binds to
   `127.0.0.1` by default for safety. To expose it:
   ```powershell
   $env:ANGEL_HOST = "0.0.0.0"; .\venv\Scripts\python.exe web_server.py
   ```
   It prints the LAN address to use. This opens Angel's endpoints (including app
   control) to everyone on the network, so only do it on a trusted Wi-Fi.

---

## HTTP API (served by `web_server.py`)

| Method | Route | Purpose |
|--------|-------|---------|
| `GET`  | `/status` | Service, LLM, and voice warm/cold state |
| `POST` | `/chat` | Send text → reply, speech text, and any command action token |
| `POST` | `/speak` | Synthesize + (optionally) play on the server; runs the pending command |
| `GET`  | `/nowplaying` | What's playing on the machine |
| `POST` | `/playback` | `pause` / `resume` / `next` / `previous` |
| `GET`  | `/audio` | Stream a generated clip |
| `POST` | `/start`, `/stop` | Launch / gracefully stop the services |

---

## Example things to say

```
hey, how's it going                 → chat, spoken reply
laugh for me                        → actually laughs (not a joke)
play blinding lights                → exact track on YouTube Music
i wanna listen to travis scott      → artist shuffle
play the after hours album          → the album
play something on apple music       → opens Apple Music
pause the music  /  skip this song  → controls whatever is playing
what song is this                   → reads back the current track
open chrome  /  close spotify       → app control (safe)
what time is it  /  tell me a joke  → utility commands
```

---

## Notes and limitations

- **Windows-only** by design (media session + `winsound`).
- `ytmusicapi` is unofficial and can break if YouTube changes its internals;
  resolution degrades to a search page rather than failing hard.
- The Windows Apple Music app ignores deep links, so Angel can only open the app
  — it can't auto-play a specific track there.
- App control is **safe mode**: graceful close only, with a system/assistant
  process denylist. There is no force-kill.
- Set `WOLFRAM_API_KEY` if you want the weather/temperature command.

---

## License

No license file is included yet. Add one before reuse or distribution.
