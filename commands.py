"""
Utility Command Handler
───────────────────────
Extracted from main.py so both the PyQt front end and the web UI dispatch the
same commands from one place. Nothing here touches audio, the mic, or Qt, which
is what lets web_server.py import it without pulling in the desktop UI stack.

Two entry points:

    plan_command(query) -> Plan
        Matches the query and returns what *would* happen. Side-effecting
        commands are returned as an unexecuted callable in Plan.action, so the
        caller decides when to fire them. The web UI uses this to hold an action
        back until audio playback starts, otherwise a browser tab opens and
        steals focus ~20s before Angel says anything about it.

    handle_command(query) -> (handled, response, mood)
        Plan and execute immediately. Original behaviour, used by main.py.

Commands split into two kinds, and the split is not cosmetic:

  * content commands (the time, calculate, wikipedia, joke, roast, speed test,
    temperature) produce the text that gets spoken, so their result is required
    before the reply can be generated. These always run inline. They are all
    read-only and none of them steal window focus, so there is nothing to defer.

  * action commands (open google/youtube, search, play, mute, volume, message,
    play music) have a side effect that is separable from the spoken
    confirmation. These are deferred.

Dependencies are resolved per command rather than all-or-nothing. The original
version gated every command behind a single import block, so one missing package
disabled the whole set — including "open google", which only needs the standard
library. Each command now checks just what it uses, and a command whose package
is missing reports that instead of silently falling through to the LLM.

`song` and `Whatsapp` are imported lazily inside their commands because both
import pywhatkit at module scope, which would otherwise make this module
unimportable in an environment that lacks it.

mood semantics:
    "positive" -> task succeeded, phrase it in character
    "negative" -> task failed, say so in character
    None       -> speak `response` as-is (jokes, roasts, wiki text)
"""

import datetime
import importlib
import operator
import os
import random
import re
import urllib.parse
import webbrowser
from typing import Callable, NamedTuple


def _optional(name):
    """Import a package, or return None if it isn't installed."""
    try:
        return importlib.import_module(name)
    except Exception:
        return None


pyautogui = _optional("pyautogui")
pywhatkit = _optional("pywhatkit")
pyjokes = _optional("pyjokes")
speedtest = _optional("speedtest")
wikipedia = _optional("wikipedia")
wolframalpha = _optional("wolframalpha")

# Kept for main.py, which imports this flag. True only when the full optional
# set is present; individual commands no longer depend on it.
FULL_VA_AVAILABLE = all(
    m is not None
    for m in (pyautogui, pywhatkit, pyjokes, speedtest, wikipedia, wolframalpha)
)

_MISSING = "{cmd} needs the {pkg} package, which isn't installed"


class Plan(NamedTuple):
    """
    The outcome of matching a query.

    handled  : True if a command matched
    response : text for the LLM to confirm, or content to speak verbatim
    mood     : "positive" | "negative" | None
    action   : deferred side effect, or None if the work is already done
    on_fail  : response text to use if `action` raises
    """
    handled: bool
    response: str | None = None
    mood: str | None = None
    action: Callable[[], None] | None = None
    on_fail: str | None = None


_NO_MATCH = Plan(handled=False)


def available() -> dict:
    """Report which optional packages are present. Used by the UI/diagnostics."""
    return {
        "pyautogui": pyautogui is not None,
        "pywhatkit": pywhatkit is not None,
        "pyjokes": pyjokes is not None,
        "speedtest": speedtest is not None,
        "wikipedia": wikipedia is not None,
        "wolframalpha": wolframalpha is not None,
        "full": FULL_VA_AVAILABLE,
    }


# ─── Browser helpers ─────────────────────────────────────────────────────────

def _google_url(term: str) -> str:
    return "https://www.google.com/search?q=" + urllib.parse.quote_plus(term)


def _youtube_url(term: str) -> str:
    return (
        "https://www.youtube.com/results?search_query="
        + urllib.parse.quote_plus(term)
    )


# ─── Wikipedia ───────────────────────────────────────────────────────────────

# Wikimedia asks that clients identify themselves; requests without a
# descriptive User-Agent get throttled or served non-JSON error pages.
_WIKI_UA = "VoiceAssistant/1.0 (local personal project)"
_WIKI_REST = "https://en.wikipedia.org/api/rest_v1/page/summary/"


def _first_sentences(text: str, count: int = 2) -> str:
    """Trim an extract to roughly `count` sentences, for a speakable length."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(parts[:count]).strip()


def _wiki_summary(topic: str) -> str | None:
    """
    Fetch a short Wikipedia summary, or None.

    Uses the REST summary endpoint rather than the `wikipedia` package. That
    package (1.4.0, last released 2014) scrapes the older action API and was
    observed here returning JSONDecodeError for every query as Wikimedia served
    it non-JSON responses. The REST endpoint is documented, stable, and returns
    clean JSON as long as a real User-Agent is sent. The package is still tried
    as a fallback since it handles fuzzy titles better.
    """
    if not topic:
        return None

    try:
        import requests
        url = _WIKI_REST + urllib.parse.quote(topic.replace(" ", "_"))
        r = requests.get(url, headers={"User-Agent": _WIKI_UA}, timeout=15)
        if r.ok:
            extract = (r.json() or {}).get("extract", "").strip()
            if extract:
                return _first_sentences(extract)
    except Exception:
        pass

    # Fallback: the package's own search-and-summarize, if it happens to work.
    if wikipedia is not None:
        try:
            return wikipedia.summary(topic, sentences=2)
        except Exception:
            pass

    return None


# ─── Wolfram Alpha ───────────────────────────────────────────────────────────

def wolfram_query(query: str):
    """Query Wolfram Alpha. Returns answer string or None."""
    if wolframalpha is None:
        return None
    try:
        client = wolframalpha.Client(
            os.environ.get("WOLFRAM_API_KEY", "YOUR_WOLFRAM_API_KEY_HERE")
        )
        res = client.query(query)
        return next(res.results).text
    except Exception:
        return None


# ─── YouTube Music ───────────────────────────────────────────────────────────

_YT_MUSIC = re.compile(r"\b(?:youtube\s+music|yt\s+music|ytmusic)\b", re.IGNORECASE)
# Explicit Apple Music request. YouTube Music stays the default; Apple is only
# used when named. "apple" alone counts, so "play X on apple" works too.
_APPLE_MUSIC = re.compile(r"\bapple(?:\s+music)?\b", re.IGNORECASE)
# Playlist request. Shuffle is the default per the user; "on normal" / "in
# order" / "no shuffle" turns it off, "shuffle"/"shuffled" makes it explicit.
_PLAYLIST = re.compile(r"\bplay\s?lists?\b", re.IGNORECASE)
_NORMAL_ORDER = re.compile(
    r"\b(?:on\s+normal|in\s+order|no\s+shuffle|without\s+shuffle|not\s+shuffled?)\b",
    re.IGNORECASE,
)


def ytmusic_home() -> str:
    """YouTube Music home. Imported lazily so a broken ytmusicapi install
    cannot stop the rest of the command set from loading."""
    try:
        import ytmusic
        return ytmusic.HOME_URL
    except Exception:
        return "https://music.youtube.com/"


# ─── Playback transport ──────────────────────────────────────────────────────
# These are matched before the play/search rules, because "play the next song"
# contains both "play" and "song" and would otherwise start a new track instead
# of skipping to the next one.

# Anchored to short command forms on purpose. An unanchored "skip|next .* song"
# also matches "play next to me by imagine dragons song", which would skip the
# current track instead of playing the requested one. Loose phrasing is handled
# by the semantic prototypes, which rewrite to these exact canonical strings.
_POLITE = r"(?:(?:yo|hey|ok|okay)\s+)?(?:can\s+you\s+|could\s+you\s+|please\s+)?"
_END = r"\s*(?:please)?\s*[?!.]*$"

_PAUSE = re.compile(
    rf"^{_POLITE}(?:pause|hold)(?:\s+the)?"
    rf"(?:\s+(?:music|song|track|it|this|playback))?{_END}"
    rf"|^{_POLITE}(?:stop|kill)(?:\s+the)?\s+(?:music|song|track|playback){_END}",
    re.IGNORECASE,
)
_RESUME = re.compile(
    rf"^{_POLITE}(?:resume|unpause|continue)(?:\s+the)?"
    rf"(?:\s+(?:music|song|track|it|playback|playing))?{_END}"
    rf"|^{_POLITE}keep\s+(?:it\s+|the\s+music\s+)?playing{_END}",
    re.IGNORECASE,
)
_NEXT = re.compile(
    rf"^{_POLITE}(?:skip|next)(?:\s+(?:this|the))?"
    rf"(?:\s+(?:song|track|one))?{_END}"
    rf"|^{_POLITE}(?:play|go\s+to)\s+the\s+next\s+(?:song|track|one){_END}",
    re.IGNORECASE,
)
_PREVIOUS = re.compile(
    rf"^{_POLITE}(?:previous|last)\s+(?:song|track){_END}"
    rf"|^{_POLITE}go\s+back\s+(?:a\s+(?:song|track)|one){_END}"
    rf"|^{_POLITE}(?:play|go\s+to)\s+the\s+(?:previous|last)\s+(?:song|track){_END}",
    re.IGNORECASE,
)
_NOW_PLAYING = re.compile(
    r"\bwhat(?:'?s|\s+is)\s+(?:this\s+|currently\s+)?(?:playing|song|track)\b"
    r"|\bwhat\s+(?:song|track)\s+is\s+(?:this|playing|on)\b"
    r"|\bwho\s+(?:sings|is\s+singing)\b"
    r"|\bname\s+of\s+(?:this|the)\s+song\b"
    r"|\bwhat\s+am\s+i\s+listening\s+to\b",
    re.IGNORECASE,
)


def _transport_plan(q: str) -> "Plan | None":
    """Map a transport phrasing onto a playback action, or None."""
    # Now playing is inline: the answer *is* the reply.
    if _NOW_PLAYING.search(q):
        import playback
        current = playback.now_playing()
        if not current:
            return Plan(True, "nothing seems to be playing right now", "negative")
        title, artist = current
        detail = f"{title} by {artist}" if artist else title
        # mood=None so this is spoken verbatim. Routing it through the LLM for
        # personality lost the answer: "what song is this" came back as "it's
        # just as catchy as you said it would be", with no title in it.
        return Plan(True, f"you're listening to {detail}", None)

    for pattern, verb, said, failed in (
        (_PAUSE, "pause", "paused the music", "couldn't pause the music"),
        (_RESUME, "resume", "started the music again", "couldn't resume the music"),
        (_NEXT, "next", "skipped to the next song", "couldn't skip the song"),
        (_PREVIOUS, "previous", "went back a song", "couldn't go back a song"),
    ):
        if not pattern.search(q):
            continue

        def _act(verb=verb):
            import playback
            if not getattr(playback, {
                "pause": "pause",
                "resume": "resume",
                "next": "next_track",
                "previous": "previous_track",
            }[verb])():
                raise RuntimeError(f"playback {verb} failed")

        return Plan(True, said, "positive", action=_act, on_fail=failed)

    return None


# ─── Laughter on request ─────────────────────────────────────────────────────
# "laugh for me" and "make me laugh" are different requests: one wants to hear
# laughter, the other wants a joke. Embeddings alone conflated them (measured
# 0.750 for "laugh for me" against the joke intent), because cosine similarity
# ignores who is doing the laughing. These patterns settle it before the
# semantic layer ever runs.

# Phrasings that mean "the joke is what I want", even though they say "laugh".
_LAUGH_IS_JOKE_REQUEST = re.compile(
    r"\b(?:make|get)\s+(?:me|us)\s+laugh(?:ing)?\b"
    r"|\bmake\s+(?:me|us)\b.*\blaugh\b"
    r"|\b(?:joke|funny|humor|humour)\b",
    re.IGNORECASE,
)

# Explicit requests to perform laughter.
_LAUGH_PERFORM = re.compile(
    r"^\s*(?:can|could|will|would)?\s*(?:you\s+)?"
    r"(?:please\s+)?(?:just\s+|maybe\s+)?"
    r"(?:laugh|giggle|chuckle)"
    r"(?:\s+(?:for\s+(?:me|us)|a\s+(?:little|bit)|out\s+loud|please|"
    r"with\s+me|now|again))*"
    r"\s*[?!.]*\s*$",
    re.IGNORECASE,
)

# Longer sentences that still clearly ask for laughter, e.g. "let me hear you
# laugh" or "do a laugh for me".
_LAUGH_PERFORM_LOOSE = re.compile(
    r"\b(?:let\s+me\s+hear\s+(?:you|your)\s+(?:laugh|giggle|chuckle)"
    r"|(?:laugh|giggle|chuckle)\s+for\s+(?:me|us)"
    r"|(?:do|give\s+me)\s+a\s+(?:laugh|giggle|chuckle))\b",
    re.IGNORECASE,
)

# Mostly laughter with a short spoken tail. A tag on its own synthesizes, but a
# few words give Turbo prosody context and keep the clip from sounding clipped.
_LAUGH_LINES = [
    "[laugh] Oh my god, stop.",
    "[laugh] There, that one's yours.",
    "[chuckle] You're ridiculous, daddy.",
    "[laugh] Happy now?",
    "[chuckle] Don't make me start again.",
]


def _wants_laughter(q: str) -> bool:
    """True when the user is asking to *hear* laughter, not to be told a joke."""
    if _LAUGH_IS_JOKE_REQUEST.search(q):
        return False
    return bool(_LAUGH_PERFORM.match(q) or _LAUGH_PERFORM_LOOSE.search(q))


# ─── Matching ────────────────────────────────────────────────────────────────

def plan_command(query: str, semantic: bool = True) -> Plan:
    """
    Match `query` against the command set without firing any side effect.

    Two passes. Literal keyword matching runs first because it costs ~0ms and
    covers explicit phrasing exactly. Only when that finds nothing does the
    semantic pass run (~25ms), rewriting loose phrasing like "I wanna watch some
    mr beast today" into "search on youtube mr beast" and matching that.

    `semantic=False` restricts it to the literal pass, which is what the
    recursive call below uses to avoid looping.
    """
    plan = _plan_literal(query)
    if plan.handled or not semantic:
        return plan
    return _plan_semantic(query)


def _plan_semantic(query: str) -> Plan:
    """
    Fall back to embedding-based intent matching.

    A hit is rewritten into canonical phrasing and fed back through the literal
    matcher, so the existing handlers stay the single source of truth for what
    each command actually does.
    """
    try:
        import intent
        import slots
    except Exception as e:
        print(f"[Commands] Semantic layer unavailable: {e}")
        return _NO_MATCH

    m = intent.match(query)
    if not m.intent:
        return _NO_MATCH

    canonical = slots.to_command(m.intent, query)
    if not canonical:
        # Intent needed an argument and none survived extraction. Treating this
        # as conversation beats running a blank search.
        print(f"[Commands] semantic {m.intent} matched but no slot; ignoring")
        return _NO_MATCH

    print(f"[Commands] semantic {m.intent} ({m.score:.3f}) "
          f"-> {canonical!r}")

    plan = _plan_literal(canonical, hint_source=query)
    if not plan.handled:
        # Canonical phrasing that the literal matcher doesn't recognise means
        # the two tables have drifted apart.
        print(f"[Commands] canonical {canonical!r} matched no literal rule")
    return plan


# ─── Open / close desktop apps ───────────────────────────────────────────────

# Web/media targets that own dedicated handlers upstream. If "open X" names one
# of these, app-control declines so the real handler (already passed) or a
# later rule takes it. Guards against "open youtube" opening a stray exe.
_APP_RESERVED = (
    "google", "youtube", "youtube music", "yt music", "apple music",
    "amazon", "wikipedia", "gmail", "classroom",
)

_OPEN_APP = re.compile(
    r"^(?:yo\s+|hey\s+|ok\s+|okay\s+)?"
    r"(?:can\s+you\s+|could\s+you\s+|please\s+)?"
    r"(?:open|launch|start|run|fire\s+up|bring\s+up|pull\s+up)\s+(.+?)"
    r"(?:\s+(?:app|application|for\s+me|please))?\s*[?!.]*$",
    re.IGNORECASE,
)
_CLOSE_APP = re.compile(
    r"^(?:yo\s+|hey\s+|ok\s+|okay\s+)?"
    r"(?:can\s+you\s+|could\s+you\s+|please\s+)?"
    r"(?:close|quit|exit|kill|shut\s+down|shut|terminate|end)\s+(.+?)"
    r"(?:\s+(?:app|application|for\s+me|please|down))?\s*[?!.]*$",
    re.IGNORECASE,
)

# App names that are actually a music request ("play ...") or a web target must
# not be swallowed here.
_APP_NOISE = re.compile(r"\b(?:tab|window|browser|song|music|playlist)\b", re.IGNORECASE)


def _clean_app_name(raw: str) -> str:
    """Trim trailing filler the regex tail did not catch."""
    name = re.sub(r"^(?:the|my|a|an)\s+", "", raw.strip(), flags=re.IGNORECASE)
    name = re.sub(r"\s+(?:app|application|window|program)\s*$", "", name,
                  flags=re.IGNORECASE)
    return name.strip()


def _is_reserved_target(name: str) -> bool:
    low = name.lower()
    return any(r in low for r in _APP_RESERVED)


def _plan_app_control(q: str, hints: str) -> "Plan | None":
    """
    Handle "open X" / "close X" for arbitrary desktop apps, or None to pass.

    Returns None (declining) when the phrasing is not an app command, names a
    reserved web target, or is really about tabs/songs — so the existing rules
    keep their behaviour.
    """
    close_m = _CLOSE_APP.match(q)
    if close_m:
        name = _clean_app_name(close_m.group(1))
        if not name or _APP_NOISE.search(name) or _is_reserved_target(name):
            return None

        # Pre-flight so the spoken reply is honest. Deferred actions run at
        # audio-start, after the confirmation is already spoken, so "closed X"
        # would be a lie if X was protected or not running. We check status
        # now (cheap, read-only) and only defer the actual terminate.
        import appcontrol
        status = appcontrol.close_status(name)
        if status == "protected":
            return Plan(
                True,
                f"I won't close {name} — it's a system or assistant process",
                "negative",
            )
        if status == "not_running":
            return Plan(True, f"{name} doesn't look like it's running",
                        "negative")

        def _close(name=name):
            import appcontrol
            ok, reason = appcontrol.close_app(name)
            if not ok:
                raise RuntimeError(reason or "couldn't close it")

        return Plan(
            True, f"closing {name}", "positive",
            action=_close, on_fail=f"couldn't close {name}",
        )

    open_m = _OPEN_APP.match(q)
    if open_m:
        name = _clean_app_name(open_m.group(1))
        if not name or _APP_NOISE.search(name) or _is_reserved_target(name):
            return None

        def _open(name=name):
            import appcontrol
            if not appcontrol.open_app(name):
                raise RuntimeError("couldn't find that app")

        return Plan(
            True, f"opened {name}", "positive",
            action=_open, on_fail=f"couldn't find {name}",
        )

    return None


def _plan_literal(query: str, hint_source: str | None = None) -> Plan:
    """
    Keyword matching.

    Match order is preserved from the original implementation; several rules
    rely on it (e.g. the bare "search" rule deliberately excludes YouTube so
    "search on youtube" reaches its own handler further down).

    `hint_source` carries the user's original wording when this is called with
    a canonical rewrite. Rewriting drops descriptor words — "play travis scott
    radio" becomes "play travis scott song" — and those words are the only
    signal for whether an album or an artist mix was wanted.
    """
    q = query.lower()
    hints = (hint_source or query).lower()

    # ── Playback transport — checked first ──
    # "play the next song" matches both this and the music rule; whichever runs
    # first wins, and skipping is what the user asked for.
    transport = _transport_plan(q)
    if transport is not None:
        return transport

    # ── System Controls (pyautogui) — deferred ──
    if "mute" in q and "youtube" not in q:
        if pyautogui is None:
            return Plan(True, _MISSING.format(cmd="muting", pkg="pyautogui"), "negative")
        return Plan(
            True, "muted the volume", "positive",
            action=lambda: pyautogui.press("volumemute"),
            on_fail="couldn't mute the volume",
        )

    if "volume up" in q:
        if pyautogui is None:
            return Plan(True, _MISSING.format(cmd="volume control", pkg="pyautogui"), "negative")
        return Plan(
            True, "turned the volume up", "positive",
            action=lambda: [pyautogui.press("volumeup") for _ in range(5)],
            on_fail="couldn't change the volume",
        )

    if "volume down" in q:
        if pyautogui is None:
            return Plan(True, _MISSING.format(cmd="volume control", pkg="pyautogui"), "negative")
        return Plan(
            True, "turned the volume down", "positive",
            action=lambda: [pyautogui.press("volumedown") for _ in range(5)],
            on_fail="couldn't change the volume",
        )

    # ── Time (inline: the result is the reply) ──
    if "the time" in q:
        t = datetime.datetime.now().strftime("%I:%M %p")
        return Plan(True, f"told the time, it's {t}", "positive")

    # ── Wikipedia (inline) ──
    if "wikipedia" in q:
        search = q.replace("wikipedia", "").strip()
        result = _wiki_summary(search)
        if result:
            return Plan(True, result, "positive")
        return Plan(True, f"couldn't find {search} on Wikipedia", "negative")

    # ── Web Search — deferred ──
    if "search" in q and "youtube" not in q:
        term = q.replace("search", "").strip()

        def _search(term=term):
            if pywhatkit is not None:
                pywhatkit.search(term)
            else:
                # pywhatkit.search just opens a Google query; the stdlib can do
                # that too, so the command works without the dependency.
                webbrowser.open(_google_url(term))

        return Plan(
            True, f"searched for {term}", "positive",
            action=_search, on_fail=f"couldn't search for {term}",
        )

    # ── YouTube Music — deferred ──
    # Checked before the YouTube rules below, which would otherwise swallow
    # "play X on youtube music" and send it to plain YouTube.
    if _YT_MUSIC.search(q) and "play" not in q:
        return Plan(
            True, "opened YouTube Music", "positive",
            action=lambda: webbrowser.open(ytmusic_home()),
            on_fail="couldn't open YouTube Music",
        )

    # ── Apple Music (just open it) — deferred ──
    if _APPLE_MUSIC.search(q) and "play" not in q and (
            "open" in q or "launch" in q or q.strip() in {"apple music", "apple"}):
        def _open_apple():
            import applemusic
            webbrowser.open(applemusic.HOME_URL)

        return Plan(
            True, "opened Apple Music", "positive",
            action=_open_apple, on_fail="couldn't open Apple Music",
        )

    # ── YouTube — deferred ──
    if "open youtube" in q and not _YT_MUSIC.search(q):
        return Plan(
            True, "opened YouTube", "positive",
            action=lambda: webbrowser.open("https://www.youtube.com/"),
            on_fail="couldn't open YouTube",
        )

    if "search on youtube" in q:
        term = q.replace("search on youtube", "").strip()
        return Plan(
            True, f"searched YouTube for {term}", "positive",
            action=lambda: webbrowser.open(_youtube_url(term)),
            on_fail=f"couldn't search YouTube for {term}",
        )

    # Videos only. A "youtube music" request is a song and belongs to the music
    # handler further down, which resolves an exact track instead.
    if "play" in q and "youtube" in q and not _YT_MUSIC.search(q):
        term = q.replace("play", "").replace("on youtube", "").strip()

        def _play(term=term):
            if pywhatkit is not None:
                pywhatkit.playonyt(term)
            else:
                # Without pywhatkit we can't autoplay the first hit, but opening
                # the results page keeps the command useful.
                webbrowser.open(_youtube_url(term))

        return Plan(
            True, f"playing {term} on YouTube", "positive",
            action=_play, on_fail=f"couldn't play {term} on YouTube",
        )

    # ── Google — deferred ──
    if "open google" in q:
        return Plan(
            True, "opened Google", "positive",
            action=lambda: webbrowser.open("https://google.com"),
            on_fail="couldn't open Google",
        )

    # ── Open / close desktop apps — deferred ──
    # Placed AFTER the dedicated web rules (open google/youtube/apple music) so
    # those specific targets keep their own handlers. Anything else that says
    # "open X" / "close X" is treated as a desktop application.
    app_plan = _plan_app_control(q, hints)
    if app_plan is not None:
        return app_plan

    # ── WhatsApp — deferred ──
    if "message" in q:
        text = q.replace("message", "").strip()
        try:
            from Whatsapp import msg
        except Exception:
            return Plan(
                True,
                _MISSING.format(cmd="messaging", pkg="pywhatkit/keyboard"),
                "negative",
            )
        return Plan(
            True, "sent the message", "positive",
            action=lambda: msg(text), on_fail="couldn't send the message",
        )

    # ── Laughter on request (inline: the laugh *is* the reply) ──
    # Checked before jokes. "laugh for me" is a request to hear laughter, while
    # "make me laugh" is a request for a joke; both contain "laugh", so the
    # directional phrasing is what separates them.
    if _wants_laughter(q):
        return Plan(True, random.choice(_LAUGH_LINES), None)

    # ── Jokes (inline: the joke *is* the reply) ──
    if "joke" in q:
        if pyjokes is None:
            return Plan(True, _MISSING.format(cmd="jokes", pkg="pyjokes"), "negative")
        return Plan(True, pyjokes.get_joke(), None)

    if "roast me" in q or "insult me" in q:
        try:
            from song import roast
        except Exception:
            return Plan(True, _MISSING.format(cmd="roasts", pkg="pywhatkit"), "negative")
        return Plan(True, roast(random.randint(1, 12)), None)

    # ── Speed Test (inline) ──
    if "speed test" in q:
        if speedtest is None:
            return Plan(True, _MISSING.format(cmd="speed test", pkg="speedtest-cli"), "negative")
        try:
            s = speedtest.Speedtest()
            s.get_best_server()
            s.download()
            s.upload()
            res = s.results.dict()
            down = int(res["download"] / 800000)
            up = int(res["upload"] / 800000)
            return Plan(True, (
                f"Download {down} Mbps, Upload {up} Mbps, "
                f"Ping {int(res['ping'])}ms"
            ), "positive")
        except Exception:
            return Plan(True, "speed test failed", "negative")

    # ── Temperature (inline) ──
    if "temperature" in q:
        words = q.split()
        place = words[-1] if words else "Kolkata"
        answer = wolfram_query(f"Temperature in {place}")
        if answer:
            return Plan(True, f"temperature in {place} is {answer}", "positive")
        return Plan(True, f"couldn't get temperature for {place}", "negative")

    # ── Calculator (inline) ──
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
            return Plan(True, f"the answer is {result}", "positive")
        except Exception:
            return Plan(True, "couldn't calculate that", "negative")

    # ── Music — deferred ──
    # Triggers on a play verb plus any music noun, OR a playlist request (which
    # need not say "song"/"music"): "play the after hours playlist".
    if "play" in q and (
            "music" in q or "song" in q
            or _PLAYLIST.search(q) or _PLAYLIST.search(hints)):
        return _plan_music(query, q, hints)

    return _NO_MATCH


def _plan_music(query: str, q: str, hints: str) -> Plan:
    """
    Route a music request to Apple Music (only when named) or the default
    YouTube Music, handling playlists and album/artist hints.

    Service and playlist detection read from `hints` (the original wording) as
    well as `q`, because a semantic match rewrites the query to the canonical
    "play <name> song", dropping the "apple music" / "playlist" words that this
    routing depends on. slots.extract cleans the name either way.
    """
    from slots import extract as _extract
    name = _extract(query)

    use_apple = bool(_APPLE_MUSIC.search(q) or _APPLE_MUSIC.search(hints))

    # ── Apple Music: open the app, user plays it themselves ──
    # The Windows Apple Music app ignores deep-link paths entirely — every URL
    # form (itmss://, music://, https track/album/search) just foregrounds the
    # app on its last screen without navigating. Verified live across every
    # scheme. Real name-based playback would need paid MusicKit, so rather than
    # pretend, we open the app and tell the user to pick the song themselves.
    # This covers songs, albums, artists, and playlists alike — none can be
    # driven by URL.
    if use_apple:
        def _open_apple():
            import applemusic
            webbrowser.open(applemusic.HOME_URL)

        # mood=None so this is spoken verbatim. Routed through the LLM it lost
        # the "you play it yourself" instruction, which is the whole point.
        return Plan(
            True,
            "I'll open Apple Music for you, but you'll have to play the song yourself",
            None,
            action=_open_apple,
            on_fail="couldn't open Apple Music",
        )

    # album/artist hints come from the original wording; the slot extractor
    # strips those descriptor words.
    prefer = None
    if re.search(r"\balbums?\b", hints):
        prefer = "album"
    elif (re.search(r"\b(?:artist|discography)\b", hints)
          or re.search(r"\bradio\s*$", hints)):
        prefer = "artist"

    def _play_song(name=name, prefer=prefer):
        import ytmusic
        ytmusic.play(name, prefer=prefer)

    label = name or "some music"
    return Plan(
        True, f"playing {label}", "positive",
        action=_play_song, on_fail=f"couldn't play {label}",
    )


# ─── Immediate execution (main.py) ───────────────────────────────────────────

def handle_command(query: str) -> tuple[bool, str | None, str | None]:
    """
    Match and run a command in one step.

    Returns (handled, response_text, mood), same contract as before the
    plan/execute split.
    """
    plan = plan_command(query)

    if plan.handled and plan.action is not None:
        try:
            plan.action()
        except Exception:
            return True, plan.on_fail or "couldn't do that", "negative"

    return plan.handled, plan.response, plan.mood
