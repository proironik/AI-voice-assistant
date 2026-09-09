"""
Playback Control
────────────────
Pause, resume, skip, and "what's playing" for whatever is actually producing
sound on this machine — YouTube Music in a browser, the Spotify app, iTunes,
anything that registers a Windows media session.

Why the Windows media session and not media keys
────────────────────────────────────────────────
Sending a media key is fire-and-forget: it cannot report what is playing, and
it goes wherever Windows decides. The session API (SMTC) exposes each player
individually, so we can read the current title and target one player. Media
keys remain as a fallback for when winsdk is missing or the session call fails.

The session we must never touch
───────────────────────────────
The assistant's own replies play through an <audio> element in the web UI, and
the browser registers that as a media session like any other. Left alone,
"pause the music" would sometimes pause the assistant mid-sentence instead of
the music. Sessions whose title matches the web UI are therefore skipped, and
dedicated music apps are preferred over browser sessions.
"""

import asyncio
from typing import NamedTuple

# Media session title the browser reports for our own page. Keep in sync with
# the <title> in web/index.html.
OWN_TITLE = "Angel — Voice Assistant"

# Browsers do not all label an <audio> element the same way: some use the page
# title, some the media URL. Any session matching one of these is Angel talking
# and must never be treated as the user's music.
OWN_TITLE_HINTS = (
    OWN_TITLE.lower(),
    "angel - voice assistant",
    "voice assistant",
    "audio?id=",
)


def _is_own(title: str) -> bool:
    low = (title or "").strip().lower()
    return any(hint in low for hint in OWN_TITLE_HINTS) if low else False

# Substrings of a session's app id that mark it as a real music player. Checked
# before browser sessions so an open YouTube Music app wins over a random tab.
MUSIC_APP_HINTS = (
    "spotify",
    "youtubemusic",
    "youtube music",
    "ytmusic",
    "itunes",
    "applemusic",
    "zune",          # Groove / Windows Media Player legacy ids
    "wmplayer",
    "musicbee",
    "foobar",
    "vlc",
)

_BROWSER_HINTS = ("chrome", "msedge", "firefox", "brave", "opera")


# Friendly names for the UI. Matched as substrings of the app id, which looks
# like "SpotifyAB.SpotifyMusic_zpdnekdrzrea0!Spotify".
_APP_NAMES = (
    ("spotify", "Spotify"),
    ("youtubemusic", "YouTube Music"),
    ("ytmusic", "YouTube Music"),
    ("itunes", "iTunes"),
    ("applemusic", "Apple Music"),
    ("msedge", "Edge"),
    ("chrome", "Chrome"),
    ("firefox", "Firefox"),
    ("brave", "Brave"),
    ("opera", "Opera"),
    ("vlc", "VLC"),
)


class NoSession(RuntimeError):
    """Raised when nothing is playing that we are willing to control."""


class Entry(NamedTuple):
    """One media session, already read so the caller needs no async."""

    session: object
    app_id: str
    title: str
    artist: str
    playing: bool

    @property
    def app_name(self) -> str:
        for hint, name in _APP_NAMES:
            if hint in self.app_id:
                return name
        # Fall back to the segment after "!", which is usually the app name.
        tail = self.app_id.split("!")[-1]
        return tail.title() if tail else ""


# ─── Windows media session (winsdk) ──────────────────────────────────────────

def _session_manager():
    from winsdk.windows.media.control import (
        GlobalSystemMediaTransportControlsSessionManager as Manager,
    )
    return Manager


async def _collect() -> list[Entry]:
    """Read every media session the system will tell us about."""
    from winsdk.windows.media.control import (
        GlobalSystemMediaTransportControlsSessionPlaybackStatus as Status,
    )

    manager = await _session_manager().request_async()
    found = []
    for session in manager.get_sessions():
        app_id = (session.source_app_user_model_id or "").lower()
        title = artist = ""
        playing = False
        try:
            props = await session.try_get_media_properties_async()
            title = props.title or ""
            artist = props.artist or ""
        except Exception:
            # A session can vanish between enumeration and read.
            pass
        try:
            playing = session.get_playback_info().playback_status == Status.PLAYING
        except Exception:
            pass
        found.append(Entry(session, app_id, title, artist, playing))
    return found


def _rank(entry: Entry) -> tuple[int, int, int]:
    """
    Lower sorts first.

    Title presence comes first, as the safety net for our own audio: title
    matching relies on the browser reporting something recognisable, so a
    titleless session must not outrank real music merely by making noise.

    Playing status comes before app type, and that ordering matters here.
    YouTube Music plays in a browser tab, so ranking dedicated apps higher
    meant a paused Spotify in the background beat the tab actually playing,
    and the buttons resumed the wrong thing.

    App type only breaks ties between sessions that are equally idle.
    """
    if any(h in entry.app_id for h in MUSIC_APP_HINTS):
        app_rank = 0
    elif any(h in entry.app_id for h in _BROWSER_HINTS):
        app_rank = 1
    else:
        app_rank = 2
    return (0 if entry.title.strip() else 1, 0 if entry.playing else 1, app_rank)


async def _pick() -> Entry:
    """Choose the session to act on, ignoring our own audio."""
    entries = [e for e in await _collect() if not _is_own(e.title)]
    if not entries:
        raise NoSession("no music session found")
    entries.sort(key=_rank)
    return entries[0]


def _run(coro):
    """Run a coroutine from a synchronous request-handler thread."""
    return asyncio.run(coro)


# ─── Public API ──────────────────────────────────────────────────────────────

def state() -> dict:
    """Everything the UI needs to render the player, never raising."""
    idle = {
        "available": False,
        "title": None,
        "artist": None,
        "app": None,
        "playing": False,
    }
    try:
        entry = _run(_pick())
    except Exception:
        # Nothing playing is the normal case, not an error worth logging on a
        # 3-second poll.
        return idle
    if not entry.title:
        return idle
    return {
        "available": True,
        "title": entry.title,
        "artist": entry.artist or None,
        "app": entry.app_name or None,
        "playing": entry.playing,
    }


def now_playing() -> tuple[str, str] | None:
    """Return (title, artist) for the active player, or None."""
    try:
        entry = _run(_pick())
    except Exception as e:
        print(f"[Playback] now_playing unavailable: {e}")
        return None
    if not entry.title:
        return None
    return entry.title, entry.artist


def _control(action: str) -> bool:
    """Send one transport command through the session API."""
    async def go():
        session = (await _pick()).session
        if action == "pause":
            return await session.try_pause_async()
        if action == "resume":
            return await session.try_play_async()
        if action == "toggle":
            return await session.try_toggle_play_pause_async()
        if action == "next":
            return await session.try_skip_next_async()
        if action == "previous":
            return await session.try_skip_previous_async()
        raise ValueError(f"unknown action {action!r}")

    try:
        return bool(_run(go()))
    except Exception as e:
        print(f"[Playback] session {action} failed ({e}); trying media key")
        return _media_key(action)


# Fallback path. Less precise: Windows routes the key itself, so it can reach
# the wrong player, but it works without winsdk.
_KEYS = {
    "pause": "playpause",
    "resume": "playpause",
    "toggle": "playpause",
    "next": "nexttrack",
    "previous": "prevtrack",
}


def _media_key(action: str) -> bool:
    key = _KEYS.get(action)
    if not key:
        return False
    try:
        import pyautogui
        pyautogui.press(key)
        return True
    except Exception as e:
        print(f"[Playback] media key {key} failed: {e}")
        return False


def pause() -> bool:
    return _control("pause")


def resume() -> bool:
    return _control("resume")


def toggle() -> bool:
    return _control("toggle")


def next_track() -> bool:
    return _control("next")


def previous_track() -> bool:
    return _control("previous")
