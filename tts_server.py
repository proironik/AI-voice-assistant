"""
Chatterbox TTS Persistent Server
────────────────────────────────
Replaces the paid Azure TTS stage with local Chatterbox-Turbo (Resemble AI, MIT).

Loads the model ONCE and stays running, accepting synthesis requests over a
local TCP socket. Cold start is ~17s, so keeping it warm matters.

IMPORTANT: run this with the dedicated TTS venv, NOT the assistant venv and NOT
the RVC venv. Chatterbox pins torch==2.6.0 while RVC needs 2.5.1+cu118, so the
environments must stay separate:

    D:\\rvc\\tts_env\\Scripts\\python.exe tts_server.py

Start this BEFORE main.py, alongside rvc_server.py.

Protocol (JSON over TCP):
    Request:  {"text": "hello", "output": "C:/path/to/tts.wav"}
              optional: "temperature", "top_p", "top_k",
                        "repetition_penalty", "reference"
    Response: {"status": "ok", "output": "...", "sr": 24000,
               "duration": 1.83, "timings": {...}}
              {"status": "error", "message": "..."}
"""

import json
import os
import re
import socket
import sys
import threading
import time
from pathlib import Path

# Line-buffer stdout. Without this, Python block-buffers when output is
# redirected and the server's progress logs stay invisible until it exits.
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except AttributeError:
    pass

# ─── Keep every cache off the C: drive ───────────────────────────────────────
# Must be set before torch / huggingface_hub are imported.
_CACHE = Path(r"D:\rvc\tts_cache")
os.environ.setdefault("HF_HOME", str(_CACHE / "hf"))
os.environ.setdefault("TMP", str(_CACHE / "tmp"))
os.environ.setdefault("TEMP", str(_CACHE / "tmp"))

import librosa
import numpy as np
import torch
import torchaudio as ta
from chatterbox.tts_turbo import ChatterboxTurboTTS

# ─── Configuration ───────────────────────────────────────────────────────────

HOST = "127.0.0.1"
PORT = 7866                      # rvc_server.py uses 7865
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Optional style reference. Chatterbox conditions both speaker and delivery;
# RVC later changes the timbre, but does not guarantee total identity removal.
REFERENCE_CLIP = Path(__file__).parent / "reference.wav"

# Turbo sampling controls verified in chatterbox-tts 0.1.7. Native tags such as
# [chuckle], [laugh], and [sigh] provide expression. cfg_weight and exaggeration
# are intentionally absent because this Turbo build explicitly ignores them.
DEFAULT_TEMPERATURE = 0.78
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 1000
DEFAULT_REPETITION_PENALTY = 1.2

# Pitch-preserving pacing. The web client normally sends 15.5; this is the
# fallback for direct clients. A higher target means less stretching and a
# shorter waveform for RVC while remaining below Turbo's usual 16-18.5 rate.
TARGET_CHARS_PER_SEC = 15.5
SPEED_MIN, SPEED_MAX = 0.65, 1.0

# ─── Model load (once) ───────────────────────────────────────────────────────

print(f"[TTS Server] Device: {DEVICE}")
if DEVICE == "cpu":
    print("[TTS Server] WARNING: running on CPU. Measured RTF was 3-6x on this")
    print("[TTS Server]          machine, i.e. slower than realtime. Expect lag.")

_t0 = time.perf_counter()
model = ChatterboxTurboTTS.from_pretrained(device=DEVICE)
print(f"[TTS Server] Chatterbox-Turbo loaded in {time.perf_counter() - _t0:.1f}s "
      f"(sr={model.sr})")

if DEVICE == "cuda":
    print(f"[TTS Server] VRAM reserved: "
          f"{torch.cuda.memory_reserved() / 1024**3:.2f}GB")

_ref = str(REFERENCE_CLIP) if REFERENCE_CLIP.exists() else None
if _ref:
    # Preparing conditionals reloads/resamples/tokenizes the reference. Doing it
    # once at startup avoids repeating that work on every synthesis request.
    _ref_t0 = time.perf_counter()
    model.prepare_conditionals(_ref, exaggeration=0.0)
    print(f"[TTS Server] Style reference cached: {REFERENCE_CLIP.name} "
          f"({time.perf_counter() - _ref_t0:.1f}s)")
else:
    print(f"[TTS Server] No {REFERENCE_CLIP.name} found — using built-in voice.")

# Generation is not thread-safe; serialize it.
_lock = threading.Lock()

# Set by a {"command": "shutdown"} request. taskkill without /F cannot reach a
# console process parked in accept(), so a hard kill was the only other way to
# stop this server; an in-protocol shutdown lets it unload the model and release
# VRAM on its own terms instead.
_shutdown = threading.Event()
_server_sock = None


def synthesize(text: str, output_wav: str, **kw) -> tuple[int, float, dict]:
    """Synthesize text and return sample rate, duration, and stage timings."""
    # A per-request reference is still supported. The normal startup reference
    # has already been cached in model.conds, so passing None reuses it.
    request_reference = kw.get("reference")

    generate_t0 = time.perf_counter()
    with _lock:
        wav = model.generate(
            text,
            audio_prompt_path=request_reference,
            temperature=float(kw.get("temperature", DEFAULT_TEMPERATURE)),
            top_p=float(kw.get("top_p", DEFAULT_TOP_P)),
            top_k=int(kw.get("top_k", DEFAULT_TOP_K)),
            repetition_penalty=float(
                kw.get("repetition_penalty", DEFAULT_REPETITION_PENALTY)
            ),
        )
    generate_ms = int((time.perf_counter() - generate_t0) * 1000)

    native_dur = wav.shape[-1] / model.sr
    # Control tokens describe audio events but are not spoken characters. If
    # counted, "[chuckle]" would make the normalizer speed the whole line up.
    visible_text = re.sub(r"\[[^\]]+\]", "", text)
    n_chars = max(len(visible_text.strip()), 1)

    speed = kw.get("speed")
    if speed is None:
        target = float(kw.get("target_chars_per_sec") or TARGET_CHARS_PER_SEC)
        native_rate = n_chars / native_dur if native_dur > 0 else target
        speed = target / native_rate if native_rate > 0 else 1.0
    speed = float(min(max(float(speed), SPEED_MIN), SPEED_MAX))

    stretch_t0 = time.perf_counter()
    if abs(speed - 1.0) > 1e-3:
        y = wav.squeeze(0).detach().cpu().numpy().astype(np.float32)
        y = librosa.effects.time_stretch(y, rate=speed)
        wav = torch.from_numpy(np.ascontiguousarray(y)).unsqueeze(0)
    stretch_ms = int((time.perf_counter() - stretch_t0) * 1000)

    final_dur = wav.shape[-1] / model.sr
    print(f"[TTS Server]   pacing: {n_chars / native_dur:.1f} -> "
          f"{n_chars / final_dur:.1f} chars/s (stretch {speed:.2f})")

    save_t0 = time.perf_counter()
    Path(output_wav).parent.mkdir(parents=True, exist_ok=True)
    ta.save(output_wav, wav, model.sr)
    save_ms = int((time.perf_counter() - save_t0) * 1000)

    return model.sr, final_dur, {
        "generate_ms": generate_ms,
        "stretch_ms": stretch_ms,
        "save_ms": save_ms,
    }


# ─── TCP Server ──────────────────────────────────────────────────────────────

def handle_client(conn, addr):
    """Handle a single synthesis request."""
    try:
        data = b""
        while True:
            chunk = conn.recv(4096)
            if not chunk:
                break
            data += chunk
            try:
                json.loads(data.decode("utf-8"))
                break          # complete JSON received
            except json.JSONDecodeError:
                continue       # keep reading

        if not data:
            return

        request = json.loads(data.decode("utf-8"))

        # ── Control commands ──
        if request.get("command") == "shutdown":
            print("[TTS Server] Shutdown requested.")
            try:
                conn.sendall(json.dumps(
                    {"status": "ok", "message": "shutting down"}
                ).encode("utf-8"))
            except Exception:
                pass
            _shutdown.set()
            # Unblocks accept() in main(), which then falls through to cleanup.
            if _server_sock is not None:
                try:
                    _server_sock.close()
                except Exception:
                    pass
            return

        if request.get("command") == "ping":
            conn.sendall(json.dumps({"status": "ok", "sr": model.sr}).encode("utf-8"))
            return

        text = request["text"]
        output_path = request["output"]

        preview = text if len(text) <= 60 else text[:57] + "..."
        print(f"[TTS Server] Synthesizing: {preview!r}")

        t0 = time.perf_counter()
        sr, duration, timings = synthesize(
            text,
            output_path,
            temperature=request.get("temperature", DEFAULT_TEMPERATURE),
            top_p=request.get("top_p", DEFAULT_TOP_P),
            top_k=request.get("top_k", DEFAULT_TOP_K),
            repetition_penalty=request.get(
                "repetition_penalty", DEFAULT_REPETITION_PENALTY
            ),
            speed=request.get("speed"),
            target_chars_per_sec=request.get("target_chars_per_sec"),
            reference=request.get("reference"),
        )
        elapsed = time.perf_counter() - t0

        conn.sendall(json.dumps({
            "status": "ok",
            "output": output_path,
            "sr": sr,
            "duration": round(duration, 3),
            "timings": timings,
        }).encode("utf-8"))

        rtf = elapsed / duration if duration else float("inf")
        print(f"[TTS Server] Done in {elapsed:.2f}s "
              f"({duration:.2f}s audio, RTF {rtf:.2f})")

    except Exception as e:
        try:
            conn.sendall(json.dumps(
                {"status": "error", "message": str(e)}
            ).encode("utf-8"))
        except Exception:
            pass
        print(f"[TTS Server] Error: {e}")
    finally:
        conn.close()


def main():
    global _server_sock

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(2)
    _server_sock = server
    print(f"[TTS Server] Listening on {HOST}:{PORT}")
    print( "[TTS Server] Ready for requests.\n")

    try:
        while not _shutdown.is_set():
            try:
                conn, addr = server.accept()
            except OSError:
                # Listener was closed by the shutdown handler.
                break
            threading.Thread(
                target=handle_client, args=(conn, addr), daemon=True
            ).start()
    except KeyboardInterrupt:
        print("\n[TTS Server] Interrupted.")
    finally:
        try:
            server.close()
        except Exception:
            pass
        if DEVICE == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        print("[TTS Server] Stopped.")


if __name__ == "__main__":
    main()
