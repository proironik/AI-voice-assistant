"""
System Audio Levels
───────────────────
Captures whatever this machine is playing and turns it into numbers the web UI
can react to: an overall level, three frequency bands, a small spectrum, and a
beat counter.

Why capture on the server at all
────────────────────────────────
A browser can only analyse audio it is playing itself. Spotify, YouTube Music
in another tab, and any desktop player are invisible to it. The alternatives
were:

  * getDisplayMedia with system audio — works, but demands a screen-share
    permission prompt on every page load.
  * A loopback input device (Stereo Mix) — often disabled, and not present on
    every machine.
  * WASAPI loopback capture, used here — no prompt, no configuration, and it
    hears every application at once.

Verified on this machine: "Speakers (Realtek(R) Audio) [Loopback]" at 48kHz
stereo, reporting RMS 0.10-0.36 and bass energy 11-37 while music played.

The catch, and how it is handled
────────────────────────────────
Loopback hears *everything*, including Angel's own replies, because her audio
comes out of the same speakers. The server cannot separate them, so the browser
ignores these frames while her audio element is playing. Suppression lives on
the client because only the client knows when she is talking.

Capture is reference-counted: the device is opened when the first browser
subscribes and closed shortly after the last one leaves, so an idle UI costs
nothing.
"""

import threading
import time
from collections import deque

# ─── Tuning ──────────────────────────────────────────────────────────────────

CHUNK = 1024              # ~21ms at 48kHz; small enough to catch a kick drum
BANDS = 12                # spectrum columns sent to the UI
IDLE_TIMEOUT = 5.0        # seconds to keep the device open after last listener

# Beat detection. A beat is bass energy standing well above its recent average,
# with a refractory gap so one kick is not counted three times.
BEAT_THRESHOLD = 1.45
BEAT_FLOOR = 0.06
BEAT_REFRACTORY = 0.22
BEAT_HISTORY = 43         # ~0.9s of frames

# Adaptive gain. System volume changes the absolute numbers by an order of
# magnitude, so levels are normalised against a slowly decaying observed peak
# rather than a fixed constant.
PEAK_DECAY = 0.9985
PEAK_FLOOR = 0.02

_state = {
    "level": 0.0,
    "bass": 0.0,
    "mid": 0.0,
    "high": 0.0,
    "bands": [0.0] * BANDS,
    "beat": 0,
    "active": False,
    "device": None,
    "error": None,
}
_lock = threading.Lock()

_subscribers = 0
_sub_lock = threading.Lock()
_thread = None
_stop = threading.Event()
_last_release = 0.0


# ─── Device discovery ────────────────────────────────────────────────────────

def _find_loopback(pyaudio_module, audio):
    """Return the loopback device matching the current default output."""
    host = audio.get_host_api_info_by_type(pyaudio_module.paWASAPI)
    speakers = audio.get_device_info_by_index(host["defaultOutputDevice"])
    if speakers.get("isLoopbackDevice"):
        return speakers
    for dev in audio.get_loopback_device_info_generator():
        if speakers["name"] in dev["name"]:
            return dev
    return None


# ─── Capture loop ────────────────────────────────────────────────────────────

def _capture():
    """Read the output device until asked to stop, publishing frames."""
    try:
        import numpy as np
        import pyaudiowpatch as pyaudio
    except Exception as e:
        with _lock:
            _state.update(active=False, error=f"capture unavailable: {e}")
        print(f"[Levels] Unavailable: {e}")
        return

    audio = pyaudio.PyAudio()
    stream = None
    try:
        device = _find_loopback(pyaudio, audio)
        if device is None:
            with _lock:
                _state.update(active=False, error="no loopback device")
            print("[Levels] No WASAPI loopback device found.")
            return

        rate = int(device["defaultSampleRate"])
        channels = int(device["maxInputChannels"])
        stream = audio.open(
            format=pyaudio.paFloat32,
            channels=channels,
            rate=rate,
            frames_per_buffer=CHUNK,
            input=True,
            input_device_index=int(device["index"]),
        )
        with _lock:
            _state.update(active=True, error=None, device=device["name"])
        print(f"[Levels] Capturing {device['name']} at {rate}Hz")

        window = np.hanning(CHUNK)
        freqs = np.fft.rfftfreq(CHUNK, 1.0 / rate)
        # Log-spaced edges: even spacing wastes most columns on treble nobody
        # can see moving.
        edges = np.geomspace(40, min(12000, rate / 2), BANDS + 1)
        band_slices = [
            (freqs >= edges[i]) & (freqs < edges[i + 1]) for i in range(BANDS)
        ]
        low = (freqs >= 20) & (freqs < 250)
        mid = (freqs >= 250) & (freqs < 2000)
        high = (freqs >= 2000) & (freqs < 8000)

        history = deque(maxlen=BEAT_HISTORY)
        peak_level = PEAK_FLOOR
        peak_band = PEAK_FLOOR
        last_beat = 0.0
        beats = 0

        while not _stop.is_set():
            # Self-release when nobody is listening. Checked here rather than in
            # a watchdog thread so no one closes the device from underneath a
            # blocking read.
            #
            # Note: WASAPI loopback delivers no buffers at all while the system
            # is completely silent, so this check is only reached once audio is
            # flowing. During silence the thread parks inside stream.read() at
            # zero CPU, which is the state we would be releasing it to anyway.
            with _sub_lock:
                nobody = _subscribers == 0
                idle_for = time.monotonic() - _last_release
            if nobody and idle_for > IDLE_TIMEOUT:
                break

            raw = stream.read(CHUNK, exception_on_overflow=False)
            samples = np.frombuffer(raw, dtype=np.float32)
            if channels > 1:
                samples = samples.reshape(-1, channels).mean(axis=1)
            if samples.size < CHUNK:
                continue

            rms = float(np.sqrt(np.mean(samples ** 2)))
            spectrum = np.abs(np.fft.rfft(samples * window)) / CHUNK

            band_energy = [float(spectrum[s].mean()) if s.any() else 0.0
                           for s in band_slices]
            bass_e = float(spectrum[low].mean())
            mid_e = float(spectrum[mid].mean())
            high_e = float(spectrum[high].mean())

            # Adaptive normalisation.
            peak_level = max(peak_level * PEAK_DECAY, rms, PEAK_FLOOR)
            peak_band = max(peak_band * PEAK_DECAY, max(band_energy), 1e-4)

            level = min(1.0, rms / peak_level)
            bands = [min(1.0, b / peak_band) for b in band_energy]
            bass = min(1.0, bass_e / peak_band)

            now = time.monotonic()
            average = sum(history) / len(history) if history else 0.0
            if (bass > BEAT_FLOOR
                    and average > 0
                    and bass > average * BEAT_THRESHOLD
                    and now - last_beat > BEAT_REFRACTORY):
                beats += 1
                last_beat = now
            history.append(bass)

            with _lock:
                _state.update(
                    level=round(level, 4),
                    bass=round(bass, 4),
                    mid=round(min(1.0, mid_e / peak_band), 4),
                    high=round(min(1.0, high_e / peak_band), 4),
                    bands=[round(b, 3) for b in bands],
                    beat=beats,
                    active=True,
                )
    except Exception as e:
        with _lock:
            _state.update(active=False, error=str(e))
        print(f"[Levels] Capture stopped: {e}")
    finally:
        if stream is not None:
            try:
                stream.close()
            except Exception:
                pass
        try:
            audio.terminate()
        except Exception:
            pass
        with _lock:
            _state.update(active=False, level=0.0, bass=0.0, mid=0.0,
                          high=0.0, bands=[0.0] * BANDS)
        print("[Levels] Capture released.")


# ─── Lifecycle ───────────────────────────────────────────────────────────────

def subscribe():
    """Register a listener, starting capture if it is not already running."""
    global _thread, _subscribers
    with _sub_lock:
        _subscribers += 1
    if _thread is None or not _thread.is_alive():
        _stop.clear()
        _thread = threading.Thread(target=_capture, daemon=True,
                                   name="audiolevels")
        _thread.start()


def unsubscribe():
    global _subscribers, _last_release
    with _sub_lock:
        _subscribers = max(0, _subscribers - 1)
        _last_release = time.monotonic()


def stop():
    """Stop capturing and release the audio device."""
    global _thread
    _stop.set()
    thread = _thread
    _thread = None
    if thread is not None and thread.is_alive():
        thread.join(timeout=2.0)


def snapshot() -> dict:
    """The most recent frame. Safe to call whether or not capture is running."""
    with _lock:
        return dict(_state)
