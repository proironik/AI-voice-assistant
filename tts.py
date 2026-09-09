"""Client for the persistent local Chatterbox-Turbo TTS server."""

import json
import re
import socket
import time
from pathlib import Path

from expressions import prepare as prepare_expression

TTS_HOST = "127.0.0.1"
TTS_PORT = 7866
SOCKET_TIMEOUT = 120
RETRIES = 2
RETRY_DELAY = 1.5

# Turbo controls that are actually consumed by the installed 0.1.7 model.
# Expression intensity is controlled with native tags such as [chuckle] and
# [sigh], not exaggeration/cfg_weight (Turbo accepts but ignores those values).
TEMPERATURE = 0.78
TOP_P = 0.95
TOP_K = 1000
REPETITION_PENALTY = 1.2

# Slightly faster than the old 14.5 chars/s setting. Native Turbo commonly runs
# at 16-18.5, so 15.5 still sounds conversational while shortening the audio
# handed to RVC and reducing the amount of time stretching required.
TARGET_CHARS_PER_SEC = 15.5


class TTSServerUnavailable(RuntimeError):
    """Raised when the Chatterbox TTS server is not reachable."""


_EMOJI = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U0001F1E6-\U0001F1FF"
    "\U00002600-\U000027BF"
    "\U0000FE00-\U0000FE0F"
    "\U00002B00-\U00002BFF"
    "\U0001F000-\U0001F2FF"
    "\U0000200D"
    "]+",
    flags=re.UNICODE,
)


def sanitize(text: str) -> str:
    """Prepare clean prose while preserving only verified native voice tags."""
    text = _EMOJI.sub("", text)
    text = prepare_expression(text).speech
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    return text


def speak(text: str, output_file: str = "tts.wav") -> str:
    """Synthesize ``text`` through the persistent server and return its path."""
    output_path = str(Path(output_file).resolve())
    text = sanitize(text)
    if not text:
        raise ValueError("Nothing to speak after sanitizing (text was empty).")

    request = json.dumps({
        "text": text,
        "output": output_path,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "repetition_penalty": REPETITION_PENALTY,
        "target_chars_per_sec": TARGET_CHARS_PER_SEC,
    }).encode("utf-8")

    last_error = None
    for attempt in range(RETRIES + 1):
        try:
            return _request(request, output_file)
        except ConnectionRefusedError as e:
            raise TTSServerUnavailable(
                f"Chatterbox TTS server not reachable on {TTS_HOST}:{TTS_PORT} ({e}). "
                "Start it with: D:\\rvc\\tts_env\\Scripts\\python.exe tts_server.py"
            ) from e
        except (ConnectionResetError, socket.timeout, OSError) as e:
            last_error = e
            if attempt < RETRIES:
                time.sleep(RETRY_DELAY)
                continue
            raise TTSServerUnavailable(
                f"TTS request failed after {RETRIES + 1} attempts: {e}"
            ) from e

    raise TTSServerUnavailable(f"TTS request failed: {last_error}")


def _request(payload: bytes, output_file: str) -> str:
    """Send one synthesis request and return the output path."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(SOCKET_TIMEOUT)
    try:
        sock.connect((TTS_HOST, TTS_PORT))
        sock.sendall(payload)
        sock.shutdown(socket.SHUT_WR)

        data = b""
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            data += chunk
    finally:
        sock.close()

    if not data:
        raise ConnectionResetError(
            "TTS server closed the connection without responding."
        )

    response = json.loads(data.decode("utf-8"))
    if response.get("status") != "ok":
        raise RuntimeError(f"TTS failed: {response.get('message')}")

    return output_file
