"""
Localhost server for the voice assistant web UI.

Serves ./web and exposes:
    GET  /status  -> which of ollama / tts / rvc are reachable
    POST /start   -> launch anything not already running (start_all.ps1)
    POST /stop    -> graceful-first shutdown of all three
    POST /chat    -> {"text": ...} -> {"reply": ..., "audio": "/audio?t=..."}
    GET  /audio   -> the most recent converted WAV

/chat runs the same pipeline main.py uses: convo.chat() for the reply,
tts.speak() for synthesis, then rvc.run_rvc() to convert timbre. TTS and RVC
are best-effort: if either is down you still get the text reply, with a note
about what was skipped, so the page stays usable while services boot.

Run:  .\\venv\\Scripts\\python.exe web_server.py
Open: http://localhost:8080
"""

import http.server
import json
import os
import select
import shutil
import socket
import sys
import threading
import time
import urllib.parse
import uuid

PORT = 8080
ROOT = os.path.dirname(os.path.abspath(__file__))
DIRECTORY = os.path.join(ROOT, "web")
AUDIO_DIR = os.path.join(ROOT, ".runtime_audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

sys.path.insert(0, ROOT)

import services  # noqa: E402  (local module, needs ROOT on sys.path first)
from expressions import prepare as prepare_expression  # noqa: E402

# TTS and RVC are single-model services. Serialize voice jobs, but no longer
# hold the browser's /chat response open while this slower work runs.
_pipeline_lock = threading.Lock()

# Request-specific audio avoids one tab overwriting final.wav before another
# tab has fetched it. Entries and files are expired opportunistically.
_audio_files = {}
_audio_lock = threading.Lock()
AUDIO_TTL = 600


def _new_audio_paths():
    audio_id = uuid.uuid4().hex
    return (
        audio_id,
        os.path.join(AUDIO_DIR, f"{audio_id}-tts.wav"),
        os.path.join(AUDIO_DIR, f"{audio_id}-final.wav"),
    )


def _remember_audio(audio_id, path):
    now = time.time()
    with _audio_lock:
        stale = [
            key for key, value in _audio_files.items()
            if now - value["created"] > AUDIO_TTL
        ]
        for key in stale:
            old = _audio_files.pop(key, None)
            if old:
                for candidate in old.get("files", ()):
                    try:
                        os.remove(candidate)
                    except OSError:
                        pass
        _audio_files[audio_id] = {
            "path": path,
            "files": (
                os.path.join(AUDIO_DIR, f"{audio_id}-tts.wav"),
                os.path.join(AUDIO_DIR, f"{audio_id}-final.wav"),
            ),
            "created": now,
        }


def _audio_path(audio_id):
    with _audio_lock:
        entry = _audio_files.get(audio_id)
    return entry["path"] if entry else None

# Commands whose side effect is held back until the browser reports that
# playback has started, so a new tab doesn't steal focus while Angel is still
# being synthesized. Keyed by a single-use token handed to the client.
_pending_actions = {}
_pending_lock = threading.Lock()
PENDING_TTL = 300  # seconds; abandoned tokens are dropped on the next sweep


def _stash_action(plan):
    """Store a deferred action and return its single-use token."""
    token = uuid.uuid4().hex
    now = time.time()
    with _pending_lock:
        # Opportunistic sweep: a client that never commits (tab closed, audio
        # blocked) would otherwise leak an entry per request.
        for stale in [
            t for t, v in _pending_actions.items() if now - v["created"] > PENDING_TTL
        ]:
            _pending_actions.pop(stale, None)
        _pending_actions[token] = {"plan": plan, "created": now}
    return token


def _take_action(token):
    """Pop an action by token. Returns None if unknown or already consumed."""
    with _pending_lock:
        entry = _pending_actions.pop(token, None)
    return entry["plan"] if entry else None


# True only while a warmup request is in flight. Whether the model is *resident*
# is asked of Ollama directly, since it can also be evicted or loaded by
# anything else on the machine.
_warming = threading.Event()
_last_warm_seconds = {"value": None}
_voice_warming = threading.Event()
_voice_ready = {"value": False}


def _warm_voice():
    """Pay the first-inference CUDA cost before the user's first spoken turn."""
    _voice_warming.set()
    _voice_ready["value"] = False
    audio_id, tts_path, final_path = _new_audio_paths()
    try:
        import tts
        import rvc

        t0 = time.perf_counter()
        with _pipeline_lock:
            tts.speak("[happy] Ready.", tts_path)
            if services.is_up("rvc"):
                rvc.run_rvc(tts_path, final_path)
        _voice_ready["value"] = True
        print(f"[Web UI] Voice pipeline warm in {time.perf_counter() - t0:.1f}s")
    except Exception as e:
        print(f"[Web UI] Voice warmup failed: {e}")
    finally:
        for candidate in (tts_path, final_path):
            try:
                os.remove(candidate)
            except OSError:
                pass
        _voice_warming.clear()


def _warm_llm():
    """Load the LLM into VRAM and build the intent index, ahead of first use."""
    _warming.set()
    try:
        import convo
        ok, secs = convo.warmup()
        _last_warm_seconds["value"] = round(secs, 1)
        print(f"[Web UI] LLM warmup {'ok' if ok else 'failed'} in {secs:.1f}s")
    except Exception as e:
        print(f"[Web UI] LLM warmup error: {e}")
    finally:
        _warming.clear()

    # Embedding the prototypes takes a couple of seconds the first time and is
    # what makes the first semantic match fast rather than slow.
    try:
        import intent
        t0 = time.perf_counter()
        if intent.warmup():
            print(f"[Web UI] Intent index/runner ready in "
                  f"{time.perf_counter() - t0:.1f}s")
    except Exception as e:
        print(f"[Web UI] Intent index error: {e}")


def _warm_after_start(include_voice: bool):
    """Warm shared-GPU stages sequentially to avoid startup contention."""
    _warm_llm()
    if include_voice:
        _warm_voice()


def _llm_state(ollama_up):
    """Report whether the next message will pay a model reload."""
    if not ollama_up:
        return {"state": "down", "seconds": _last_warm_seconds["value"]}
    if _warming.is_set():
        return {"state": "warming", "seconds": _last_warm_seconds["value"]}
    try:
        import convo
        loaded = convo.is_loaded()
    except Exception:
        loaded = False
    return {
        "state": "warm" if loaded else "cold",
        "seconds": _last_warm_seconds["value"],
    }


def _run_action(plan):
    """
    Execute a deferred action. Returns (ran_ok, warning_or_None).

    The spoken confirmation was generated before this ran, so a failure here
    means Angel already claimed success. Surfacing the warning in the UI is the
    honest signal that the words and the outcome disagreed.
    """
    try:
        plan.action()
        print("[Web UI]   action executed")
        return True, None
    except Exception as e:
        warning = f"{plan.on_fail or 'Action failed'}: {e}"
        print(f"[Web UI]   action failed: {e}")
        return False, warning


def _lan_ip():
    """Best-effort LAN IP to print for the phone. Never raises."""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # No packets are sent; this just picks the interface that would route
        # to an external address, i.e. the real LAN IP rather than 127.0.0.1.
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "0.0.0.0"
    finally:
        s.close()


def _play_wav_blocking(path):
    """
    Play a WAV through the laptop's own speakers and return when it finishes.

    Used for remote clients (the phone) that cannot play audio themselves: the
    voice must come from the computer, so the server plays it. winsound is in
    the stdlib and needs no extra dependency; SND_FILENAME is synchronous so
    the caller can commit a deferred action exactly when playback ends, the
    same ordering the browser gets from its <audio> element.
    """
    try:
        import winsound
        winsound.PlaySound(path, winsound.SND_FILENAME)
        return True
    except Exception as e:
        print(f"[Web UI]   server-side playback failed: {e}")
        return False


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    # ─── Routing ────────────────────────────────────────────────────────────

    def do_GET(self):
        if self.path == "/status":
            self._handle_status()
        elif self.path == "/nowplaying":
            self._handle_now_playing()
        elif self.path == "/levels":
            self._handle_levels()
        elif self.path.split("?")[0] == "/audio":
            self._handle_audio()
        else:
            super().do_GET()

    def do_POST(self):
        route = self.path.split("?")[0]
        if route == "/start":
            self._handle_start()
        elif route == "/stop":
            self._handle_stop()
        elif route == "/chat":
            self._handle_chat()
        elif route == "/speak":
            self._handle_speak()
        elif route == "/commit":
            self._handle_commit()
        elif route == "/playback":
            self._handle_playback()
        else:
            self.send_error(404)

    # ─── Helpers ────────────────────────────────────────────────────────────

    def _send_json(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self):
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    # ─── GET /status ────────────────────────────────────────────────────────

    def _handle_status(self):
        st = services.status()
        # Ollama's port opens before any model is resident, so "up" alone
        # doesn't tell you whether the next message will pay a ~24s load.
        st["llm"] = _llm_state(st["services"]["ollama"])
        if not st["services"]["tts"]:
            voice_state = "down"
        elif _voice_warming.is_set():
            voice_state = "warming"
        elif _voice_ready["value"]:
            voice_state = "warm"
        else:
            voice_state = "cold"
        st["voice"] = {"state": voice_state}
        self._send_json(st)

    # ─── GET /nowplaying, POST /playback ────────────────────────────────────

    def _handle_now_playing(self):
        """Whatever is playing on this machine, for the UI's media player."""
        try:
            import playback
            self._send_json(playback.state())
        except Exception as e:
            # Polled every few seconds; a failure here must not surface as an
            # error in the UI, just an idle player.
            self._send_json({
                "available": False, "title": None, "artist": None,
                "app": None, "playing": False, "message": str(e),
            })

    def _handle_playback(self):
        """Transport control from the UI buttons."""
        try:
            body = self._read_json_body()
        except Exception:
            self._send_json({"status": "error", "message": "bad request"}, 400)
            return

        action = str(body.get("action") or "")
        try:
            import playback
        except Exception as e:
            self._send_json({"status": "error", "message": str(e)}, 500)
            return

        # An explicit table rather than getattr(playback, action): the action
        # arrives from a request body and must not be able to name arbitrary
        # module attributes.
        actions = {
            "toggle": playback.toggle,
            "pause": playback.pause,
            "resume": playback.resume,
            "next": playback.next_track,
            "previous": playback.previous_track,
        }
        handler = actions.get(action)
        if handler is None:
            self._send_json(
                {"status": "error", "message": f"unknown action {action!r}"}, 400
            )
            return

        try:
            ok = handler()
        except Exception as e:
            self._send_json({"status": "error", "message": str(e)}, 500)
            return

        print(f"[Web UI] playback {action} -> {'ok' if ok else 'refused'}")
        self._send_json({
            "status": "ok" if ok else "error",
            "message": "" if ok else (
                "Nothing is playing, or that player refuses remote control."
            ),
            "now_playing": playback.state(),
        })

    # ─── GET /levels ────────────────────────────────────────────────────────

    # Server-sent events at a fixed rate. One long-lived connection rather than
    # a request per frame: 30 polls a second would be absurd, and the browser
    # cannot analyse other applications' audio itself.
    LEVELS_FPS = 30

    def _client_gone(self) -> bool:
        """
        True once the browser has closed the connection.

        A write alone will not tell us: these frames are small, so they sit in
        the socket buffer and the failure only surfaces once it fills, which
        measured well over ten seconds. The capture device stayed open that
        whole time with nobody listening. Peeking for EOF spots a clean FIN
        immediately.
        """
        try:
            readable, _, _ = select.select([self.connection], [], [], 0)
            if not readable:
                return False
            return not self.connection.recv(1, socket.MSG_PEEK)
        except (OSError, ValueError):
            return True

    def _handle_levels(self):
        try:
            import audiolevels
        except Exception as e:
            self._send_json({"status": "error", "message": str(e)}, 500)
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        # Buffering a stream would defeat the point.
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        audiolevels.subscribe()
        interval = 1.0 / self.LEVELS_FPS
        # Deadline-based pacing. Sleeping a flat interval after each write let
        # serialisation cost accumulate and the stream ran at ~21fps instead of
        # the requested 30.
        next_frame = time.monotonic()
        try:
            while True:
                if self._client_gone():
                    break
                frame = audiolevels.snapshot()
                payload = json.dumps({
                    "level": frame["level"],
                    "bass": frame["bass"],
                    "mid": frame["mid"],
                    "high": frame["high"],
                    "bands": frame["bands"],
                    "beat": frame["beat"],
                    "active": frame["active"],
                })
                self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                self.wfile.flush()
                next_frame += interval
                time.sleep(max(0.0, next_frame - time.monotonic()))
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            # Normal: the tab was closed or reloaded.
            pass
        except Exception as e:
            print(f"[Web UI] /levels ended: {e}")
        finally:
            audiolevels.unsubscribe()

    # ─── GET /audio ─────────────────────────────────────────────────────────

    def _handle_audio(self):
        """Stream request-specific generated audio without loading it all into RAM."""
        query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        audio_id = (query.get("id") or [""])[0]
        path = _audio_path(audio_id)
        if not path or not os.path.exists(path):
            self.send_error(404, "Audio not found or expired")
            return

        try:
            size = os.path.getsize(path)
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(size))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            with open(path, "rb") as source:
                shutil.copyfileobj(source, self.wfile, length=64 * 1024)
        except OSError as e:
            if not self.wfile.closed:
                print(f"[Web UI] Could not serve audio: {e}")

    # ─── POST /chat ─────────────────────────────────────────────────────────

    def _handle_chat(self):
        """Resolve text immediately; voice generation happens in POST /speak."""
        request_t0 = time.perf_counter()
        try:
            text = (self._read_json_body().get("text") or "").strip()
            if not text:
                self._send_json({"status": "error", "message": "empty text"}, 400)
                return

            import convo

            print(f"[Web UI] > {text}")
            result = {
                "status": "ok",
                "reply": None,
                "speech": None,
                "expression": [],
                "command": False,
                "action": None,
                "timings": {},
                "warnings": [],
            }

            raw_reply = self._resolve_reply(text, convo, result)
            expressive = prepare_expression(raw_reply)
            result["reply"] = expressive.display
            result["speech"] = expressive.speech
            result["expression"] = list(expressive.tags)
            result["timings"]["text_total_ms"] = int(
                (time.perf_counter() - request_t0) * 1000
            )
            print(f"[Web UI] < {expressive.display}")
            if expressive.tags:
                print(f"[Web UI]   voice expression: {', '.join(expressive.tags)}")

            # The browser can display this response now, then call /speak while
            # the user is already reading it. Previously /chat remained blocked
            # through TTS and RVC, hiding the text for several extra seconds.
            self._send_json(result)

        except Exception as e:
            print(f"[Web UI] /chat error: {e}")
            self._send_json({"status": "error", "message": str(e)}, 500)

    def _resolve_reply(self, text, convo, result):
        """
        Produce the text to speak, deferring any side effect.

        Mirrors main.py's AssistantWorker: try the command set first, and only
        fall through to the LLM when nothing matched. A matched command with a
        mood still goes through the LLM so the confirmation stays in character;
        a matched command without one (jokes, roasts, wiki text) is spoken
        verbatim and skips the LLM entirely.

        Uses plan_command rather than handle_command so an action command is
        matched but not yet run; the token in result["action"] is what the
        browser posts back to /commit once audio starts.
        """
        plan = None
        try:
            from commands import plan_command
            t0 = time.perf_counter()
            plan = plan_command(text)
            if plan.handled:
                result["timings"]["command_ms"] = int((time.perf_counter() - t0) * 1000)
        except Exception as e:
            # A broken command shouldn't swallow the turn; fall back to chat.
            msg = f"Command handler error: {e}"
            print(f"[Web UI] {msg}")
            result["warnings"].append(msg)

        handled = bool(plan and plan.handled)

        if handled:
            result["command"] = True
            deferred = plan.action is not None
            print(f"[Web UI]   command matched -> {plan.response!r} "
                  f"(mood={plan.mood}, deferred={deferred})")
            if deferred:
                result["action"] = _stash_action(plan)
            if plan.mood is None:
                return plan.response  # speak as-is, no LLM round trip

        t0 = time.perf_counter()
        reply = (
            convo.chat(plan.response, mood=plan.mood) if handled
            else convo.chat(text)
        )
        result["timings"]["llm_ms"] = int((time.perf_counter() - t0) * 1000)
        return reply

    # ─── POST /speak ────────────────────────────────────────────────────────

    def _handle_speak(self):
        """Generate TTS/RVC after the text reply has already reached the UI.

        Two client modes:
          * Browser (default): returns the /audio URL and the browser plays it,
            then posts /commit itself when playback starts.
          * Remote / phone ("play": true): the voice must come from the laptop,
            not the phone, so the server plays the WAV through its own speakers
            and — since it now knows exactly when playback finishes — runs any
            pending command action itself. The phone never plays audio and does
            not need to call /commit.
        """
        try:
            body = self._read_json_body()
            text = (body.get("text") or "").strip()
            if not text:
                self._send_json({"status": "error", "message": "empty text"}, 400)
                return

            play_here = bool(body.get("play"))
            token = (body.get("token") or "").strip()

            audio_id, tts_path, final_path = _new_audio_paths()
            result = {"status": "ok", "audio": None, "timings": {}, "warnings": []}

            queue_t0 = time.perf_counter()
            with _pipeline_lock:
                result["timings"]["voice_queue_ms"] = int(
                    (time.perf_counter() - queue_t0) * 1000
                )
                audio_path = self._synthesize(
                    text, result, tts_path=tts_path, final_path=final_path
                )

            if audio_path:
                _remember_audio(audio_id, audio_path)
                result["audio"] = f"/audio?id={audio_id}"

                if play_here:
                    # Everything happens on the computer: play the voice here,
                    # then fire the deferred command when it starts. Playback is
                    # blocking, so commit right before it (action fires as the
                    # audio begins, matching the browser's "on play" timing),
                    # then block until it ends so timings stay honest.
                    result["played_on_server"] = True
                    if token:
                        plan = _take_action(token)
                        if plan:
                            ran, warn = _run_action(plan)
                            result["ran"] = ran
                            if warn:
                                result["warnings"].append(warn)
                    _play_wav_blocking(audio_path)
            else:
                for candidate in (tts_path, final_path):
                    try:
                        os.remove(candidate)
                    except OSError:
                        pass

            self._send_json(result)
        except Exception as e:
            print(f"[Web UI] /speak error: {e}")
            self._send_json({"status": "error", "message": str(e)}, 500)

    # ─── POST /commit ───────────────────────────────────────────────────────

    def _handle_commit(self):
        """
        Run a deferred command action. Called by the browser once playback of
        the reply has actually started.

        Tokens are single-use, so a duplicate commit (a replayed 'play' event,
        say) is a no-op rather than opening a second tab.
        """
        try:
            token = (self._read_json_body().get("token") or "").strip()
            plan = _take_action(token)
            if not plan:
                # Unknown or already consumed. Not an error worth surfacing.
                self._send_json({"status": "ok", "ran": False, "reason": "no pending action"})
                return

            ok, err = _run_action(plan)
            self._send_json({
                "status": "ok",
                "ran": ok,
                "warning": err,
            })

        except Exception as e:
            print(f"[Web UI] /commit error: {e}")
            self._send_json({"status": "error", "message": str(e)}, 500)

    def _synthesize(self, reply, result, tts_path, final_path):
        """Run TTS then optional RVC, returning the generated audio path."""
        import tts
        import rvc

        try:
            t0 = time.perf_counter()
            tts.speak(reply, tts_path)
            result["timings"]["tts_ms"] = int((time.perf_counter() - t0) * 1000)
        except Exception as e:
            msg = f"TTS unavailable: {e}"
            print(f"[Web UI] {msg}")
            result["warnings"].append(msg)
            return None

        audio_path = tts_path

        # Probe only RVC. The old full status() call also opened Ollama and TTS
        # connections in the hot path even though their state was irrelevant.
        if services.is_up("rvc"):
            try:
                t0 = time.perf_counter()
                rvc.run_rvc(tts_path, final_path)
                result["timings"]["rvc_ms"] = int((time.perf_counter() - t0) * 1000)
                audio_path = final_path
            except Exception as e:
                msg = f"RVC conversion failed, using raw TTS voice: {e}"
                print(f"[Web UI] {msg}")
                result["warnings"].append(msg)
        else:
            result["warnings"].append(
                "RVC server is down — playing raw TTS voice (not converted)."
            )

        result["converted"] = audio_path == final_path
        return audio_path

    # ─── POST /start ────────────────────────────────────────────────────────

    def _handle_start(self):
        print("[Web UI] Starting services...")
        try:
            result = services.start_all()

            # Pull mistral into VRAM now, in the background. Ollama reports the
            # port as ready before any model is loaded, so without this the
            # first message still pays the ~24s load.
            launched = set(result.get("launched") or [])
            warm_voice = bool(launched.intersection({"tts", "rvc"}))
            if result.get("services", {}).get("ollama"):
                threading.Thread(
                    target=_warm_after_start,
                    args=(warm_voice,),
                    daemon=True,
                ).start()
                result["message"] = (
                    (result.get("message") or "") + " Warming up the model..."
                ).strip()
                if warm_voice:
                    result["message"] += " Warming up voice..."

            self._send_json(result)
        except Exception as e:
            self._send_json({"status": "error", "message": str(e)}, 500)

    # ─── POST /stop ─────────────────────────────────────────────────────────

    def _handle_stop(self):
        print("[Web UI] Stopping services...")
        try:
            self._send_json(services.stop_all())
        except Exception as e:
            self._send_json({"status": "error", "message": str(e)}, 500)

    def log_message(self, fmt, *args):
        # Quiet the per-request access log; keep our own prints readable.
        request = str(args[0]) if args else ""
        if "/status" in request or "/nowplaying" in request or "/levels" in request:
            return
        super().log_message(fmt, *args)

    def handle_one_request(self):
        try:
            super().handle_one_request()
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            # A tab closed or reloaded mid-request. Routine with a long-lived
            # /levels stream, and socketserver would otherwise dump a full
            # traceback for each one.
            self.close_connection = True


def main():
    # Bind address. Loopback-only by default so the server is not exposed to
    # the network unless asked. The phone app needs LAN access, which is an
    # explicit opt-in: set ANGEL_HOST=0.0.0.0 (or a specific LAN IP). Keeping
    # localhost the default means nothing on the network can reach the
    # assistant — including its app-control and command endpoints — by accident.
    host = os.environ.get("ANGEL_HOST", "127.0.0.1")

    # Threaded: /status polling must not queue behind a long /chat call.
    server = http.server.ThreadingHTTPServer((host, PORT), Handler)
    if host in ("0.0.0.0", "::"):
        lan_ip = _lan_ip()
        print(f"[Web UI] Serving on http://{lan_ip}:{PORT} "
              f"(LAN — reachable by the phone app)")
        print("[Web UI] WARNING: exposed to the local network. Anyone on this "
              "Wi-Fi can reach Angel's endpoints, including app control.")
    else:
        print(f"[Web UI] Serving on http://localhost:{PORT} "
              f"(localhost only — set ANGEL_HOST=0.0.0.0 for the phone app)")
    st = services.status()
    for name, ok in st["services"].items():
        print(f"[Web UI]   {st['labels'][name]:<16} {'up' if ok else 'down'}")

    # A web-server restart can happen while Ollama/TTS/RVC stay alive. Warm the
    # prompt and semantic runner here too; otherwise the first user turn after
    # that restart pays 15-20s of prompt-cache setup despite the model being
    # resident in GPU memory.
    if st["services"]["ollama"]:
        threading.Thread(target=_warm_after_start, args=(False,), daemon=True).start()

    print("[Web UI] Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Web UI] Shutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
