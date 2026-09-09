"""
Semantic Intent Matching
────────────────────────
Maps natural phrasing onto the command set, so "I wanna watch some mr beast
today" reaches the same handler as the literal "search on youtube mr beast".

Why embeddings rather than an LLM classifier
────────────────────────────────────────────
An LLM classification pass would cost a second generation call, and concurrent
calls do not overlap on this machine — measured 6.06s sequential vs 6.52s wall
for two parallel requests, with ~146MB of VRAM free, so Ollama cannot run a
second slot. A classifier call would therefore add its full latency on top of
the reply.

Embeddings avoid that. all-minilm is 76MB, sits in VRAM alongside mistral
without evicting it, and scores a query against every prototype in ~25ms once
the connection is warm.

How it works
────────────
Each intent carries example phrasings ("prototypes"). At load time every
prototype is embedded once. A query is embedded, cosine-compared against all of
them, and the best match wins if two conditions hold:

    absolute   : top score >= THRESHOLD
    separation : top score exceeds the best *other* intent by MARGIN

The margin matters. "shut up for a second" scored 0.410 for mute against 0.408
for the runner-up — statistically a coin flip. Requiring separation turns those
near-ties into "no match", which falls through to ordinary conversation rather
than firing the wrong action.

`chitchat` is a real intent with its own prototypes, not an absence of one.
Without it, affectionate lines like "say something sweet" get forced toward
whichever command happens to be least dissimilar.

Slot extraction is deliberately not done here; see slots.py.

Two corrections to pure scoring
───────────────────────────────
1. Media requests are routed by rule, before any embedding runs. Cosine
   similarity against short prototypes collapses once a query carries an
   entity, and the collapse lands right on the threshold:

       "show me gta vi trailer"       0.409  -> below threshold, nothing ran
       "show me the gta vi trailer"   0.423  -> matched, searched YouTube
       "play gta 6 trailer"           0.372  -> below threshold, nothing ran
       "pull up gta 6 trailer"        0.314  -> ranked volume_up top

   One article decided whether the assistant acted at all, which is why the
   same request sometimes played a video, sometimes searched, and sometimes
   just talked. Verbs and media-type words do not dilute, so they decide.

2. For intents that normally carry an entity, the query is also scored with
   the entity masked out ("show me gta vi trailer" -> "show me something",
   0.409 -> 0.696) and the better score wins. Masking is deliberately not
   applied to no-argument intents such as volume_up, because "pull up
   something" scores 0.396 against it and would start firing volume keys.
"""

import math
import re

import requests

import slots

# 127.0.0.1, not "localhost" — see the note in convo.py. Resolving "localhost"
# costs ~2s per new connection on this machine.
OLLAMA_HOST = "http://127.0.0.1:11434"
EMBED_URL = f"{OLLAMA_HOST}/api/embed"
EMBED_MODEL = "all-minilm"
KEEP_ALIVE = "30m"

# Tuned against the measurements in this module's docstring. Scores for correct
# matches on loose phrasing ran 0.44-0.90; the observed false-positive risk was
# all in the sub-0.42 near-tie band.
THRESHOLD = 0.42
MARGIN = 0.03

_session = requests.Session()


# ─── Intent prototypes ───────────────────────────────────────────────────────
# Each key maps to the canonical command phrasing that commands.plan_command
# already understands, so a match can be rewritten into a string it accepts.

PROTOTYPES = {
    "youtube_search": [
        "search on youtube",
        "i wanna watch something on youtube",
        "put on a video",
        "find me a video",
        "i feel like watching something",
        "lets watch something",
        "i wanna watch some videos",
        "show me a video",
    ],
    "youtube_play": [
        "play this on youtube",
        "play the video",
        "start playing a video on youtube",
    ],
    "open_youtube": [
        "open youtube",
        "launch youtube",
        "go to youtube",
        "bring up youtube",
    ],
    "open_google": [
        "open google",
        "launch google",
        "bring up google",
        "go to google",
    ],
    "web_search": [
        "search for something",
        "look this up",
        "google this for me",
        "find information about this",
        "look up something for me",
    ],
    "play_music": [
        "play some music",
        "put on a song",
        "i wanna listen to music",
        "play me a track",
        "i feel like listening to something",
        "play this on youtube music",
        "put on some tunes",
        # Apple Music and playlist phrasings resolve to the same intent; the
        # command handler decides service and playlist vs track from the words.
        "play this on apple music",
        "play a song on apple music",
        "play the playlist",
        "shuffle the playlist",
        "play a playlist on apple music",
    ],
    # Transport control. These reach whatever is actually playing through the
    # Windows media session, so they are not YouTube Music specific.
    "pause_music": [
        "pause the music",
        "pause the song",
        "stop the music",
        "hold the music",
    ],
    "resume_music": [
        "resume the music",
        "unpause the music",
        "keep playing",
        "continue the song",
    ],
    "next_track": [
        "skip this song",
        "next song",
        "skip this track",
        "play the next song",
    ],
    "previous_track": [
        "previous song",
        "go back a song",
        "play the last song",
    ],
    # "track" pulls hard toward play_music's "play me a track" prototype
    # (measured 0.662 against 0.434), so the question forms are spelled out.
    "now_playing": [
        "what song is this",
        "what track is this",
        "whats playing right now",
        "who sings this song",
        "what am i listening to",
        "whats this song called",
        "name this track",
    ],
    "volume_up": [
        "turn the volume up",
        "make it louder",
        "raise the volume",
        "crank it up",
    ],
    "volume_down": [
        "turn the volume down",
        "make it quieter",
        "lower the volume",
        "turn it down a bit",
    ],
    "mute": [
        "mute the volume",
        "mute it",
        "silence the audio",
    ],
    "the_time": [
        "what time is it",
        "tell me the time",
        "whats the time right now",
    ],
    "wikipedia": [
        "look this up on wikipedia",
        "what does wikipedia say about this",
        "tell me about this topic",
    ],
    "joke": [
        "tell me a joke",
        "say something funny",
        "make me laugh",
        "got any good jokes",
        "hit me with a joke",
    ],
    # Kept separate from `joke` on purpose. "laugh for me" scored 0.750 against
    # the joke prototypes, so asking to hear laughter was answered with a joke.
    # Cosine similarity cannot see who is meant to be laughing; giving the
    # performance request its own prototypes is what restores the distinction.
    "perform_laugh": [
        "laugh for me",
        "can you laugh",
        "just laugh",
        "laugh a little",
        "let me hear you laugh",
        "giggle for me",
        "chuckle for me",
        "do a laugh for me",
        "laugh out loud for me",
    ],
    "speed_test": [
        "run a speed test",
        "how fast is my internet",
        "check my connection speed",
    ],
    # Conversation is an intent in its own right. Without these prototypes,
    # affectionate or idle chatter drifts toward whichever command is least
    # dissimilar and fires it.
    "chitchat": [
        "hey you up",
        "how are you feeling",
        "i missed you",
        "say something sweet",
        "what are you wearing",
        "tell me you love me",
        "im bored talk to me",
        "what are you thinking about",
        "good morning babe",
        "i had a rough day",
        "do you love me",
        "youre so pretty",
        "what do you think",
        "whats your opinion",
    ],
}

# Intents that mean "just talk to me" — matching one is the same as no match.
CONVERSATIONAL = {"chitchat"}


# ─── Deterministic media routing ─────────────────────────────────────────────
# Runs before embeddings, so a media request always resolves the same way. It
# also keeps working when the embedding model is unavailable.

def _words(*terms: str) -> re.Pattern:
    """Word-boundary alternation. Boundaries matter: a bare 'play' substring
    also appears inside 'display', and 'see' inside 'seen'."""
    return re.compile(r"\b(?:" + "|".join(terms) + r")\b", re.IGNORECASE)


# A named piece of video content. Asking for one means "play it", not "show me
# a list of results".
_VIDEO_WORDS = _words(
    "trailer", "teaser", "video", "clip", "clips", "episode", "gameplay",
    "walkthrough", "review", "tutorial", "documentary", "short", "shorts",
    "scene", "highlights", "podcast", "movie", "film", "reveal",
)
_MUSIC_WORDS = _words(
    "song", "songs", "music", "track", "tracks", "album", "playlist",
    "soundtrack", "ost",
)
# "music video" contains a music word but is a video request.
_MUSIC_VIDEO = re.compile(r"\bmusic\s+video\b|\bmv\b", re.IGNORECASE)

# Watching leans toward browsing ("watch some mr beast" -> results page).
_WATCH_VERBS = _words("show", "watch", "see", "pull up", "bring up", "stream")
# Playing leans toward audio unless a video word says otherwise.
_PLAY_VERBS = _words("play", "put on", "throw on", "listen to", "hear")

# "next song" / "last track" is transport. Deliberately requires the noun, so a
# title such as "next to me" is still routed as something to play.
_NEXT_OR_LAST = re.compile(
    r"\b(?:next|last|previous)\s+(?:song|track|one)\b", re.IGNORECASE
)

def route_media(query: str) -> str | None:
    """
    Resolve an entity-bearing media request to an intent, or None.

    Deliberately conservative: it requires a title-like slot plus either a
    media-type word or a watch/play verb. Anything else returns None and falls
    through to embedding scoring, then to conversation.
    """
    q = (query or "").lower()
    if slots.TRANSPORT_WORDS.search(q) or _NEXT_OR_LAST.search(q):
        # About the current track, not a new one. Let the transport rules and
        # scoring handle it.
        return None

    slot = slots.extract(query)
    if not slots.looks_like_title(slot):
        return None
    if not slots.names_something(slot):
        # Nothing is actually named, so this is not "play <thing>". Scoring can
        # tell "play some music" from "start the music up again"; a keyword rule
        # cannot.
        return None

    is_music_video = bool(_MUSIC_VIDEO.search(q))
    if _VIDEO_WORDS.search(q) or is_music_video:
        # A named video: autoplay the first hit instead of a results page.
        return "youtube_play"
    if _MUSIC_WORDS.search(q):
        return "play_music"
    if _WATCH_VERBS.search(q):
        # No media-type word, so the entity is more likely a channel or topic
        # ("watch some mr beast") than one specific video. Results page.
        return "youtube_search"
    if _PLAY_VERBS.search(q):
        # "play despacito" — bare play verb, treat as music. The music handler
        # already falls back to YouTube when the local library misses.
        return "play_music"
    return None


# ─── Embedding ───────────────────────────────────────────────────────────────

def _embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch of strings. Batching is ~6ms/text vs ~25ms per call."""
    r = _session.post(
        EMBED_URL,
        json={"model": EMBED_MODEL, "input": texts, "keep_alive": KEEP_ALIVE},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["embeddings"]


def _cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


# ─── Index ───────────────────────────────────────────────────────────────────

_index = {"labels": [], "vectors": [], "ready": False, "error": None}


def build_index(force: bool = False) -> bool:
    """
    Embed every prototype once. Safe to call repeatedly.

    Returns True if the index is usable. A failure here (embedding model not
    pulled, Ollama down) is not fatal: match() then reports no match and the
    caller falls back to keyword matching plus plain conversation.
    """
    if _index["ready"] and not force:
        return True

    labels, phrases = [], []
    for label, examples in PROTOTYPES.items():
        for phrase in examples:
            labels.append(label)
            phrases.append(phrase)

    try:
        vectors = _embed(phrases)
    except Exception as e:
        _index["error"] = str(e)
        _index["ready"] = False
        print(f"[Intent] Index build failed: {e}")
        return False

    _index.update(labels=labels, vectors=vectors, ready=True, error=None)
    print(f"[Intent] Indexed {len(phrases)} prototypes "
          f"across {len(PROTOTYPES)} intents")
    return True


def ready() -> bool:
    return _index["ready"]


def warmup() -> bool:
    """Keep both the prototype index and the tiny embedding runner hot."""
    if not build_index():
        return False
    try:
        _embed(["warm semantic routing"])
        return True
    except Exception as e:
        print(f"[Intent] Warmup failed: {e}")
        return False


# ─── Matching ────────────────────────────────────────────────────────────────

class Match:
    """An intent decision, with the scores that produced it."""

    def __init__(self, intent, score=0.0, runner_up=None, runner_up_score=0.0,
                 reason=""):
        self.intent = intent               # None when nothing matched
        self.score = score
        self.runner_up = runner_up
        self.runner_up_score = runner_up_score
        self.reason = reason

    def __repr__(self):
        return (f"Match(intent={self.intent!r}, score={self.score:.3f}, "
                f"runner_up={self.runner_up!r}:{self.runner_up_score:.3f}, "
                f"reason={self.reason!r})")

    def as_dict(self):
        return {
            "intent": self.intent,
            "score": round(self.score, 3),
            "runner_up": self.runner_up,
            "runner_up_score": round(self.runner_up_score, 3),
            "reason": self.reason,
        }


# Intents whose queries normally carry an entity, and therefore suffer entity
# dilution. Only these are eligible for masked scoring; extending it to
# no-argument intents makes "pull up something" match volume_up at 0.396.
SLOT_INTENTS = {
    "youtube_search",
    "youtube_play",
    "web_search",
    "play_music",
    "wikipedia",
}


def _carrier(query: str, slot: str) -> str | None:
    """
    Rebuild the query with the entity replaced by a neutral word.

    "show me gta vi trailer" -> "show me something", which scores against the
    prototypes on its phrasing alone. Returns None when there is no frame left
    to score (the slot was the entire query).
    """
    if not slot:
        return None
    low = query.lower()
    i = low.find(slot)
    if i < 0:
        return None
    carrier = re.sub(r"\s+", " ", query[:i] + "something" + query[i + len(slot):]).strip()
    if carrier.lower() in {low.strip(), "something"}:
        return None
    return carrier


def match(query: str) -> Match:
    """
    Classify `query`. Returns Match(intent=None, ...) when it should be treated
    as ordinary conversation.
    """
    query = (query or "").strip()
    if not query:
        return Match(None, reason="empty query")

    # Rule-based media routing first: it is deterministic, costs ~0ms, and does
    # not depend on the embedding model being reachable.
    forced = route_media(query)
    if forced:
        return Match(forced, 1.0, reason="media rule")

    if not _index["ready"] and not build_index():
        return Match(None, reason=f"index unavailable: {_index['error']}")

    # Masking is only meaningful when the residual really is an entity. Without
    # this guard "watch out" reduces to "watch something" and matches
    # youtube_search, searching for the leftover word "out".
    slot = slots.extract(query)
    carrier = _carrier(query, slot) if slots.looks_like_title(slot) else None
    try:
        variants = [query] + ([carrier] if carrier else [])
        embedded = _embed(variants)
    except Exception as e:
        return Match(None, reason=f"embed failed: {e}")

    q = embedded[0]
    q_carrier = embedded[1] if carrier else None

    # Best score per intent, so an intent with many prototypes isn't favoured.
    best_per_intent = {}
    masked = set()
    for label, vec in zip(_index["labels"], _index["vectors"]):
        s = _cosine(q, vec)
        from_mask = False
        if q_carrier is not None and label in SLOT_INTENTS:
            s_masked = _cosine(q_carrier, vec)
            if s_masked > s:
                s = s_masked
                from_mask = True
        if s > best_per_intent.get(label, -1.0):
            best_per_intent[label] = s
            masked.discard(label)
            if from_mask:
                masked.add(label)

    ranked = sorted(best_per_intent.items(), key=lambda kv: kv[1], reverse=True)
    top, top_score = ranked[0]
    runner, runner_score = (ranked[1] if len(ranked) > 1 else (None, 0.0))

    if top_score < THRESHOLD:
        return Match(None, top_score, runner, runner_score,
                     f"below threshold ({top_score:.3f} < {THRESHOLD})")

    if top_score - runner_score < MARGIN:
        # Too close to call; guessing here is how you get a browser tab opening
        # when the user only wanted to chat.
        return Match(None, top_score, runner, runner_score,
                     f"ambiguous: {top}:{top_score:.3f} vs "
                     f"{runner}:{runner_score:.3f}")

    if top in CONVERSATIONAL:
        return Match(None, top_score, runner, runner_score, "conversational")

    return Match(top, top_score, runner, runner_score,
                 "matched (entity masked)" if top in masked else "matched")
