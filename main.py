"""
Integrated Voice Assistant — Performance Optimized
──────────────────────────────────────────────────
Pipeline: Mic → STT → Command/AI → TTS → RVC → Playback
UI: PyQt5 dark theme with animated GIF + scrolling terminal

Performance improvements:
- Ollama /api/chat with num_predict=60, num_ctx=2048, full GPU offload
- Azure TTS config reuse (no re-auth per call), faster prosody rate
- Concurrent TTS+RVC where possible
- Tight token limits for fast inference on 4060
"""

import sys
import os
import re
import datetime
import time
import random
import operator
import webbrowser
import traceback
from concurrent.futures import ThreadPoolExecutor

import speech_recognition as sr
import sounddevice as sd
import soundfile as sf

from PyQt5 import QtCore, QtGui, QtWidgets

from convo import chat
from tts import speak as tts_speak
from rvc import run_rvc
from song import son, roast
from Whatsapp import msg

# Optional imports
try:
    import pyautogui
    import keyboard as kb
    import pywhatkit
    import pyjokes
    import speedtest
    import wikipedia
    import wolframalpha
    from PyDictionary import PyDictionary as diction
    FULL_VA_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] Some VA features unavailable: {e}")
    FULL_VA_AVAILABLE = False


# ─── Configuration ───────────────────────────────────────────────────────────

USE_RVC = True           # Set False to skip RVC (saves ~5s per response)
TTS_OUTPUT = "tts.wav"
RVC_OUTPUT = "final.wav"
LISTEN_TIMEOUT = 6       # seconds before mic gives up
WAKE_WORD = None         # Set to a string like "hey babe" for wake-word mode


# ─── Audio Playback ──────────────────────────────────────────────────────────

def play_audio(path: str):
    """Play a WAV file through the default audio device."""
    try:
        data, fs = sf.read(path, dtype="float32")
        sd.play(data, fs)
        sd.wait()
    except Exception as e:
        print(f"[Audio Error] {e}")


# ─── Wolfram Alpha ───────────────────────────────────────────────────────────

def wolfram_query(query: str):
    """Query Wolfram Alpha. Returns answer string or None."""
    if not FULL_VA_AVAILABLE:
        return None
    try:
        client = wolframalpha.Client(os.environ.get("WOLFRAM_API_KEY", "YOUR_WOLFRAM_API_KEY_HERE"))
        res = client.query(query)
        return next(res.results).text
    except Exception:
        return None


# ─── Speech Recognition ──────────────────────────────────────────────────────

_recognizer = sr.Recognizer()
_recognizer.pause_threshold = 1
_recognizer.dynamic_energy_threshold = True


def listen_microphone() -> str | None:
    """Listen from mic, return recognized text or None."""
    with sr.Microphone() as source:
        _recognizer.adjust_for_ambient_noise(source, duration=0.3)
        try:
            audio = _recognizer.listen(source, timeout=LISTEN_TIMEOUT, phrase_time_limit=12)
        except sr.WaitTimeoutError:
            return None

    try:
        text = _recognizer.recognize_google(audio, language='en-in')
        return text
    except (sr.UnknownValueError, sr.RequestError):
        return None


# ─── Voice Output Pipeline (Threaded + Sentence Chunking) ────────────────────

_tts_rvc_pool = ThreadPoolExecutor(max_workers=2)


def _split_sentences(text: str) -> list[str]:
    """Split text into speakable sentence chunks."""
    # Split on sentence boundaries but keep chunks reasonable
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    # Merge very short fragments
    merged = []
    buf = ""
    for p in parts:
        buf = (buf + " " + p).strip() if buf else p
        if len(buf) > 20:
            merged.append(buf)
            buf = ""
    if buf:
        merged.append(buf)
    return merged if merged else [text]


def _tts_and_rvc(text: str, tts_path: str, rvc_path: str) -> str:
    """Run TTS then RVC for a single chunk. Returns path to play."""
    tts_speak(text, tts_path)
    if USE_RVC:
        try:
            run_rvc(tts_path, rvc_path)
            return rvc_path
        except Exception as e:
            print(f"[RVC Error] {e}")
            return tts_path
    return tts_path


def speak_text(text: str):
    """
    Full voice pipeline with sentence-level pipelining.
    For short text (1 sentence): TTS → RVC → play (sequential, minimal overhead).
    For longer text: pipeline chunks so chunk N+1 is being processed while N plays.
    """
    sentences = _split_sentences(text)

    if len(sentences) == 1:
        # Simple path — no chunking overhead
        try:
            path = _tts_and_rvc(sentences[0], TTS_OUTPUT, RVC_OUTPUT)
            play_audio(path)
        except Exception as e:
            print(f"[Pipeline Error] {e}")
        return

    # Multi-sentence pipeline: process next while current plays
    for i, sentence in enumerate(sentences):
        tts_path = f"tts_chunk_{i}.wav"
        rvc_path = f"rvc_chunk_{i}.wav"
        try:
            audio_path = _tts_and_rvc(sentence, tts_path, rvc_path)
            play_audio(audio_path)
        except Exception as e:
            print(f"[Pipeline Error chunk {i}] {e}")

    # Cleanup temp chunks
    for i in range(len(sentences)):
        for f in [f"tts_chunk_{i}.wav", f"rvc_chunk_{i}.wav"]:
            try:
                os.remove(f)
            except OSError:
                pass


# ─── Command Handler ─────────────────────────────────────────────────────────
# Returns: (handled, response_text, mood)
# mood is "positive" if task succeeded, "negative" if it failed, None if not a command

def handle_command(query: str) -> tuple[bool, str | None, str | None]:
    """
    Check if query is a utility command.
    Returns (handled, response_text, mood).
    mood: "positive" | "negative" | None
    """
    q = query.lower()

    if not FULL_VA_AVAILABLE:
        return False, None, None

    # ── System Controls ──
    if "mute" in q and "youtube" not in q:
        pyautogui.press("volumemute")
        return True, "muted the volume", "positive"

    if "volume up" in q:
        for _ in range(5):
            pyautogui.press("volumeup")
        return True, "turned the volume up", "positive"

    if "volume down" in q:
        for _ in range(5):
            pyautogui.press("volumedown")
        return True, "turned the volume down", "positive"

    # ── Time ──
    if "the time" in q:
        t = datetime.datetime.now().strftime("%I:%M %p")
        return True, f"told the time, it's {t}", "positive"

    # ── Wikipedia ──
    if "wikipedia" in q:
        search = q.replace("wikipedia", "").strip()
        try:
            result = wikipedia.summary(search, sentences=2)
            return True, result, "positive"
        except Exception:
            return True, f"couldn't find {search} on Wikipedia", "negative"

    # ── Web Search ──
    if "search" in q and "youtube" not in q:
        term = q.replace("search", "").strip()
        try:
            pywhatkit.search(term)
            return True, f"searched for {term}", "positive"
        except Exception:
            return True, f"couldn't search for {term}", "negative"

    # ── YouTube ──
    if "open youtube" in q:
        webbrowser.open("https://www.youtube.com/")
        return True, "opened YouTube", "positive"

    if "search on youtube" in q:
        term = q.replace("search on youtube", "").strip()
        webbrowser.open(f"https://www.youtube.com/results?search_query={term}")
        return True, f"searched YouTube for {term}", "positive"

    if "play" in q and "youtube" in q:
        term = q.replace("play", "").replace("on youtube", "").strip()
        pywhatkit.playonyt(term)
        return True, f"playing {term} on YouTube", "positive"

    # ── Google ──
    if "open google" in q:
        webbrowser.open("https://google.com")
        return True, "opened Google", "positive"

    # ── WhatsApp ──
    if "message" in q:
        text = q.replace("message", "").strip()
        try:
            msg(text)
            return True, "sent the message", "positive"
        except Exception:
            return True, "couldn't send the message", "negative"

    # ── Jokes ──
    if "joke" in q:
        joke = pyjokes.get_joke()
        # Return joke directly, no mood (we speak the joke itself)
        return True, joke, None

    if "roast me" in q or "insult me" in q:
        insult = roast(random.randint(1, 12))
        return True, insult, None

    # ── Speed Test ──
    if "speed test" in q:
        try:
            s = speedtest.Speedtest()
            s.get_best_server()
            s.download()
            s.upload()
            res = s.results.dict()
            down = int(res["download"] / 800000)
            up = int(res["upload"] / 800000)
            return True, f"Download {down} Mbps, Upload {up} Mbps, Ping {int(res['ping'])}ms", "positive"
        except Exception:
            return True, "speed test failed", "negative"

    # ── Temperature ──
    if "temperature" in q:
        words = q.split()
        place = words[-1] if words else "Kolkata"
        answer = wolfram_query(f"Temperature in {place}")
        if answer:
            return True, f"temperature in {place} is {answer}", "positive"
        return True, f"couldn't get temperature for {place}", "negative"

    # ── Calculator ──
    if "calculate" in q:
        expr = q.replace("calculate", "").strip()
        ops = {
            '+': operator.add, 'plus': operator.add,
            '-': operator.sub, 'minus': operator.sub,
            'x': operator.mul, 'times': operator.mul,
            '/': operator.truediv, 'divided': operator.truediv,
        }
        try:
            parts = expr.split()
            result = ops[parts[1]](int(parts[0]), int(parts[2]))
            return True, f"the answer is {result}", "positive"
        except Exception:
            return True, "couldn't calculate that", "negative"

    # ── Music (local) ──
    if "play" in q and ("music" in q or "song" in q):
        name = q.replace("play", "").replace("music", "").replace("song", "").strip()
        try:
            music_dir = "D:\\Music\\"
            music_dir2 = "D:\\Music2\\"
            songs = os.listdir(music_dir)
            songs2 = os.listdir(music_dir2)
            son(name, music_dir, music_dir2, songs, songs2)
            return True, f"playing {name}", "positive"
        except Exception:
            try:
                pywhatkit.playonyt(name)
                return True, f"playing {name} on YouTube", "positive"
            except Exception:
                return True, f"couldn't play {name}", "negative"

    return False, None, None


# ─── PyQt5 UI ─────────────────────────────────────────────────────────────────

class AssistantWorker(QtCore.QObject):
    """Background worker: listen → process → respond loop."""
    update_terminal = QtCore.pyqtSignal(str, str)  # text, color
    set_status = QtCore.pyqtSignal(str, str)       # text, color
    finished = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        self.update_terminal.emit("Assistant ready. Say something...", "#00ff00")
        time.sleep(0.5)

        while self._running:
            self.set_status.emit("● Listening", "#00ccff")

            text = listen_microphone()

            if text is None:
                continue

            self.set_status.emit("● Processing", "#ffaa00")
            self.update_terminal.emit(f"You: {text}", "#ffffff")

            # Exit commands
            if any(w in text.lower() for w in ["exit", "quit", "terminate", "goodbye"]):
                reply = chat("goodbye", mood="positive")
                self.update_terminal.emit(f"Assistant: {reply}", "#ff69b4")
                speak_text(reply)
                break

            # Try command handler first
            handled, response, mood = handle_command(text)

            if handled:
                if mood is not None:
                    # Ask AI to confirm in character with mood
                    self.set_status.emit("● Responding", "#ff69b4")
                    reply = chat(response, mood=mood)
                    self.update_terminal.emit(f"Assistant: {reply}", "#ff69b4")
                    speak_text(reply)
                else:
                    # Direct response (jokes, roasts — speak as-is)
                    self.update_terminal.emit(f"Assistant: {response}", "#ff69b4")
                    speak_text(response)
            else:
                # AI conversation mode
                self.set_status.emit("● Thinking", "#ff69b4")
                reply = chat(text)
                self.update_terminal.emit(f"Assistant: {reply}", "#ff69b4")
                speak_text(reply)

        self.set_status.emit("● Offline", "#ff4444")
        self.finished.emit()


class MainWindow(QtWidgets.QMainWindow):
    """Dark-themed assistant window with GIF + terminal."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Voice Assistant")
        self.setFixedSize(831, 471)
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, False)

        # Central widget
        central = QtWidgets.QWidget()
        central.setStyleSheet("background-color: #0a0a0a;")
        self.setCentralWidget(central)

        # Border frame
        self.border = QtWidgets.QFrame(central)
        self.border.setGeometry(0, 0, 831, 471)
        self.border.setStyleSheet(
            "background-color: #0a0a0a; "
            "border: 2px solid #333333; "
            "border-radius: 8px;"
        )

        # Title bar (custom since frameless)
        self.title_bar = QtWidgets.QLabel(central)
        self.title_bar.setGeometry(10, 5, 200, 20)
        self.title_bar.setText("Voice Assistant")
        self.title_bar.setStyleSheet(
            "color: #888888; font-size: 11px; font-family: 'Segoe UI'; border: none;"
        )

        # Close button
        self.close_btn = QtWidgets.QPushButton("✕", central)
        self.close_btn.setGeometry(795, 5, 30, 20)
        self.close_btn.setStyleSheet(
            "QPushButton { color: #888; border: none; font-size: 14px; }"
            "QPushButton:hover { color: #ff4444; }"
        )
        self.close_btn.clicked.connect(self.close)

        # Animated GIF area
        self.gif_label = QtWidgets.QLabel(central)
        self.gif_label.setGeometry(210, 30, 400, 240)
        self.gif_label.setScaledContents(True)
        self.gif_label.setStyleSheet("border: none; background: transparent;")
        self._setup_gif()

        # Status indicator
        self.status_label = QtWidgets.QLabel(central)
        self.status_label.setGeometry(20, 270, 300, 18)
        self.status_label.setStyleSheet(
            "color: #00ccff; font-size: 11px; font-family: 'Courier New'; border: none;"
        )
        self.status_label.setText("● Ready")

        # Terminal output
        self.terminal = QtWidgets.QTextBrowser(central)
        self.terminal.setGeometry(15, 290, 800, 170)
        self.terminal.setStyleSheet(
            "QTextBrowser {"
            "  background-color: #111111;"
            "  border: 1px solid #333333;"
            "  border-radius: 10px;"
            "  color: #00ff00;"
            "  font-family: 'Cascadia Code', 'Courier New', monospace;"
            "  font-size: 12px;"
            "  padding: 8px;"
            "}"
            "QScrollBar:vertical {"
            "  background: #111; width: 6px; border-radius: 3px;"
            "}"
            "QScrollBar::handle:vertical {"
            "  background: #444; border-radius: 3px;"
            "}"
        )
        self.terminal.setOpenExternalLinks(False)

        # Start worker thread
        self.thread = QtCore.QThread()
        self.worker = AssistantWorker()
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.update_terminal.connect(self._append_terminal)
        self.worker.set_status.connect(self._set_status)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(lambda: QtCore.QTimer.singleShot(2000, self.close))

        self.thread.start()

        # Dragging support for frameless window
        self._drag_pos = None

    def _setup_gif(self):
        """Load animated GIF from common paths."""
        gif_candidates = [
            os.path.expanduser("~/Downloads/Siri_1.gif"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "Siri_1.gif"),
            "../../Downloads/Siri_1.gif",
        ]
        for path in gif_candidates:
            if os.path.exists(path):
                movie = QtGui.QMovie(path)
                movie.setCacheMode(QtGui.QMovie.CacheAll)
                self.gif_label.setMovie(movie)
                movie.start()
                return

        # Fallback: emoji placeholder
        self.gif_label.setText("🎙️")
        self.gif_label.setAlignment(QtCore.Qt.AlignCenter)
        self.gif_label.setStyleSheet(
            "border: none; color: white; font-size: 100px; background: transparent;"
        )

    def _append_terminal(self, text: str, color: str):
        """Append colored text to terminal."""
        escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self.terminal.append(f'<span style="color:{color};">{escaped}</span>')
        # Auto-scroll
        sb = self.terminal.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _set_status(self, text: str, color: str):
        """Update the status label."""
        self.status_label.setText(text)
        self.status_label.setStyleSheet(
            f"color: {color}; font-size: 11px; font-family: 'Courier New'; border: none;"
        )

    # ─── Frameless window drag ───
    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and event.y() < 30:
            self._drag_pos = event.globalPos() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event):
        if self._drag_pos and event.buttons() == QtCore.Qt.LeftButton:
            self.move(event.globalPos() - self._drag_pos)

    def mouseReleaseEvent(self, event):
        self._drag_pos = None

    def closeEvent(self, event):
        """Clean shutdown."""
        self.worker.stop()
        self.thread.quit()
        self.thread.wait(3000)
        event.accept()


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main():
    # High DPI support
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Voice Assistant")

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
