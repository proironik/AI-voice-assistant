"""
YouTube Music Resolution
────────────────────────
Turns "blinding lights" into an exact YouTube Music track URL so a song request
opens the song itself rather than a page of search results.

Why ytmusicapi and not an official API
──────────────────────────────────────
YouTube Music has no official public playback API, and the alternatives were
worse for this setup:

  * Apple MusicKit    — needs a paid Apple Developer membership.
  * Spotify Web API   — player endpoints require Premium on the app owner's
                        account, and development-mode apps cap at 5 listeners.
  * ytmusicapi        — no key, no login, no cost for catalog search.

A YouTube Premium subscription already covers ad-free playback and background
play of whatever URL we open, so search is the only piece that was missing.

The cost of that choice is that ytmusicapi rides YouTube's internal endpoints
and can break when Google changes them. Nothing here raises on failure; each
step degrades to a less precise one:

    1. exact track   music.youtube.com/watch?v=<id>    resolved by ytmusicapi
    2. search page   music.youtube.com/search?q=<name> library up, no match
    3. plain YouTube pywhatkit.playonyt / results page  library missing/broken

Resolution deliberately happens inside the deferred command action, not while
the reply is being written, so a slow lookup never delays what the user reads.
"""

import re
import threading
import unicodedata
import urllib.parse
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

HOME_URL = "https://music.youtube.com/"
WATCH_URL = "https://music.youtube.com/watch?v={}"
SEARCH_URL = "https://music.youtube.com/search?q={}"
# watch?list= starts playback immediately; playlist?list= only opens the page.
LIST_URL = "https://music.youtube.com/watch?list={}"
CHANNEL_URL = "https://music.youtube.com/channel/{}"

# ytmusicapi has no per-request timeout, so the call is run in a worker thread
# and abandoned if it overruns. Playback then falls back to a deep link. The
# measured warm search was well under a second; this only catches hangs.
RESOLVE_DEADLINE = 6.0

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ytmusic")
_client = None
_client_lock = threading.Lock()
_client_failed = False


@dataclass(frozen=True)
class Track:
    """One resolved YouTube Music track."""

    title: str
    artist: str
    video_id: str

    @property
    def url(self) -> str:
        return WATCH_URL.format(self.video_id)

    @property
    def label(self) -> str:
        return f"{self.title} by {self.artist}" if self.artist else self.title


@dataclass(frozen=True)
class Target:
    """A resolved thing to open, whatever kind it turned out to be."""

    url: str
    kind: str            # "artist" | "album" | "track"
    label: str


@dataclass(frozen=True)
class Playback:
    """What was actually opened, for logging and reply text."""

    url: str
    source: str          # "artist"|"album"|"track"|"search"|"youtube"|"home"
    label: str
    track: Track | None = None


def _get_client():
    """Return a cached unauthenticated YTMusic client, or None."""
    global _client, _client_failed
    if _client is not None or _client_failed:
        return _client
    with _client_lock:
        if _client is None and not _client_failed:
            try:
                from ytmusicapi import YTMusic
                _client = YTMusic()
                print("[YTMusic] Client ready (unauthenticated search)")
            except Exception as e:
                # Missing package, or ytmusicapi's internals moved. Either way
                # this stays a fallback, never an error the user hears about.
                _client_failed = True
                print(f"[YTMusic] Unavailable, will fall back to YouTube: {e}")
    return _client


def available() -> bool:
    """True when exact track resolution is possible."""
    return _get_client() is not None


def search_url(name: str) -> str:
    return SEARCH_URL.format(urllib.parse.quote_plus(name))


# YouTube Music always returns *something*: a nonsense query answered with
# "Ocean Sounds: 2 Hours of Relaxing Ocean Waves". Autoplaying that is worse
# than showing a search page, so a match has to share a word with the request.
_MIN_TOKEN = 3


def _tokens(text: str) -> set[str]:
    """Lowercase, accent-folded words of 3+ characters.

    Folding matters for real titles: "senorita" would otherwise miss
    "Señorita" and lose an exact match that the user clearly wanted.
    """
    plain = unicodedata.normalize("NFKD", text or "")
    plain = "".join(c for c in plain if not unicodedata.combining(c))
    return {w for w in re.findall(r"[a-z0-9]+", plain.lower()) if len(w) >= _MIN_TOKEN}


# Words that describe the request rather than name anything in it.
_DESCRIPTORS = {
    "album", "albums", "song", "songs", "track", "tracks", "radio",
    "music", "playlist", "discography", "mix", "stuff",
}


def _is_relevant(query: str, title: str, artist: str) -> bool:
    wanted = _tokens(query)
    if not wanted:
        # Short queries like "u2" carry no usable token; trust the ranking.
        return True
    return bool(wanted & _tokens(f"{title} {artist}"))


def _names_the_same_thing(query: str, name: str) -> bool:
    """
    True when `query` is effectively the whole name of `name`.

    Artist search is loose: searching "blinding lights" returns The Weeknd as
    the top artist, so preferring artist results blindly would turn every song
    request into a shuffle of its performer. Requiring the query to account for
    the entire artist name keeps "travis scott" an artist while "blinding
    lights" and "travis scott sicko mode" stay songs.

    One extra word in the name is tolerated so "weeknd" still matches "The
    Weeknd" and "eilish" matches "Billie Eilish".

    Descriptor words are ignored on the request side. Position-anchored filler
    stripping cannot remove an interior "album", so "the after hours album by
    the weeknd" still arrives with it attached and would otherwise fail to
    match the album "After Hours".
    """
    wanted, have = _tokens(query) - _DESCRIPTORS, _tokens(name)
    if not wanted or not have:
        return False
    if wanted == have:
        return True
    return wanted <= have and len(have - wanted) <= 1


def _search_artist(name: str) -> Target | None:
    """Resolve an artist to a shuffle of their catalogue."""
    client = _get_client()
    if client is None:
        return None

    for item in client.search(name, filter="artists", limit=3) or []:
        found = item.get("artist") or item.get("title") or ""
        if not _names_the_same_thing(name, found):
            continue
        # shuffleId shuffles this artist's own songs; radioId is a broader
        # artist radio that mixes in similar acts. "I wanna listen to travis
        # scott" means his catalogue, so shuffle wins when both exist.
        list_id = item.get("shuffleId") or item.get("radioId")
        if list_id:
            return Target(LIST_URL.format(list_id), "artist", found)
        browse_id = item.get("browseId")
        if browse_id:
            # No mix available: the artist page still beats a search page.
            return Target(CHANNEL_URL.format(browse_id), "artist", found)
    return None


def _search_album(name: str) -> Target | None:
    """Resolve an album to its playlist."""
    client = _get_client()
    if client is None:
        return None

    for item in client.search(name, filter="albums", limit=5) or []:
        title = item.get("title") or ""
        artists = ", ".join(
            a.get("name", "") for a in (item.get("artists") or []) if a.get("name")
        )
        # Album titles are heavily duplicated ("After Hours" belongs to at
        # least a dozen acts), so accept a hit on either the title or the
        # title-plus-artist reading of the request.
        if not (_names_the_same_thing(name, title)
                or _names_the_same_thing(name, f"{title} {artists}")):
            continue
        playlist_id = item.get("playlistId")
        if playlist_id:
            label = f"{title} by {artists}" if artists else title
            return Target(LIST_URL.format(playlist_id), "album", label)
    return None


def _search(name: str) -> Track | None:
    """Blocking search for the closest song match."""
    client = _get_client()
    if client is None:
        return None

    results = client.search(name, filter="songs", limit=5)
    for item in results or []:
        video_id = item.get("videoId")
        if not video_id:
            # Some rows (episodes, unavailable regions) carry no playable id.
            continue
        artists = ", ".join(
            a.get("name", "") for a in (item.get("artists") or []) if a.get("name")
        )
        title = item.get("title") or name
        if not _is_relevant(name, title, artists):
            continue
        return Track(title=title, artist=artists, video_id=video_id)
    return None


def _search_song_target(name: str) -> Target | None:
    track = _search(name)
    return None if track is None else Target(track.url, "track", track.label)


def _safe(step, name: str) -> Target | Track | None:
    """Run one resolution step, treating any failure as "no result"."""
    try:
        return step(name)
    except Exception as e:
        print(f"[YTMusic] {step.__name__} failed for {name!r}: {e}")
        return None


def _resolve_target(name: str, prefer: str | None) -> Target | None:
    """
    Work out what `name` refers to and how to play it.

    An explicit hint short-circuits the guessing. Otherwise the catalogue
    decides, and the order is the whole trick:

    A name-match against artists is not sufficient evidence on its own.
    YouTube Music lists artist entries named after hit songs — searching
    artists for "blinding lights" returns an artist called "Blinding Lights",
    and "sicko mode" returns one called "SICKO MODE". Preferring artists first
    therefore shuffled a knock-off channel instead of playing the song.

    So a song whose *title* accounts for the whole request wins first. That
    leaves "travis scott" to the artist branch, because his top song is titled
    "FE!N", not "Travis Scott".
    """
    if prefer == "album":
        return (_safe(_search_album, name)
                or _safe(_search_artist, name)
                or _safe(_search_song_target, name))
    if prefer == "artist":
        return _safe(_search_artist, name) or _safe(_search_song_target, name)

    song = _safe(_search, name)
    if song is not None and _names_the_same_thing(name, song.title):
        return Target(song.url, "track", song.label)

    artist = _safe(_search_artist, name)
    if artist is not None:
        return artist

    # Not a clean song title and not a known artist: a genre or mood request
    # like "some lofi". The best song hit is a better answer than a page.
    return None if song is None else Target(song.url, "track", song.label)


def resolve(name: str, prefer: str | None = None,
            deadline: float = RESOLVE_DEADLINE) -> Target | None:
    """Resolve `name` to something playable, or None if it cannot be done."""
    name = (name or "").strip()
    if not name or not available():
        return None
    try:
        return _executor.submit(_resolve_target, name, prefer).result(timeout=deadline)
    except Exception as e:
        print(f"[YTMusic] Search failed for {name!r}: {e}")
        return None


def play(name: str, prefer: str | None = None, open_url=webbrowser.open) -> Playback:
    """
    Open `name` on YouTube Music, degrading through the fallback chain.

    `prefer` nudges resolution when the request said so explicitly ("album",
    "artist"). `open_url` is injectable so tests can exercise every branch
    without launching a browser.
    """
    name = (name or "").strip()

    if not name:
        # "play some music" with nothing named. With Premium the home feed
        # starts a personalized mix, which is a better answer than a search
        # for an empty string.
        open_url(HOME_URL)
        return Playback(HOME_URL, "home", "your YouTube Music mix")

    target = resolve(name, prefer)
    if target is not None:
        open_url(target.url)
        print(f"[YTMusic] Resolved {name!r} -> {target.kind}: {target.label}")
        return Playback(target.url, target.kind, target.label)

    if available():
        # The library works but had no playable match: show it in YouTube Music
        # rather than dropping the user back to plain YouTube.
        url = search_url(name)
        open_url(url)
        print(f"[YTMusic] No exact match for {name!r}, opened search")
        return Playback(url, "search", name)

    return _fallback_youtube(name, open_url)


def _fallback_youtube(name: str, open_url=webbrowser.open) -> Playback:
    """Last resort: plain YouTube, which needs no extra dependency."""
    try:
        import pywhatkit
        # playonyt opens the first hit directly. It drives a browser itself, so
        # open_url is not used on this path.
        pywhatkit.playonyt(name)
        print(f"[YTMusic] Fell back to YouTube autoplay for {name!r}")
        return Playback("", "youtube", name)
    except Exception as e:
        url = ("https://www.youtube.com/results?search_query="
               + urllib.parse.quote_plus(name))
        open_url(url)
        print(f"[YTMusic] pywhatkit unavailable ({e}); opened YouTube results")
        return Playback(url, "youtube", name)
