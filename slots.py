"""
Slot Extraction
───────────────
Recovers the argument from a natural utterance: "I wanna watch some mr beast
today" -> "mr beast".

Embeddings classify the *intent* but discard word order and position, so they
cannot tell you which span of the sentence is the search term. That needs a
separate step.

Approach: iterative prefix and suffix stripping against curated filler lists,
rather than removing filler words wherever they appear. Position-restricted
stripping is what keeps meaningful interior words intact — deleting "the"
globally would turn "the office" into "office", and deleting "play" globally
would break "play that funky music". Anchoring to the ends avoids both.

An LLM could extract slots more accurately, but it would cost a full generation
call (~0.7-3s, and it cannot overlap with the reply on this GPU), which defeats
the point of using embeddings for speed.
"""

import re

# Stripped from the front, longest first, repeatedly until nothing matches.
PREFIX_FILLER = [
    # requests / politeness
    "can you please", "could you please", "would you please",
    "can you", "could you", "would you", "will you", "please",
    "hey babe", "hey", "yo", "ok", "okay", "alright",
    # intent phrasing
    "i wanna watch", "i want to watch", "i wanna see", "i want to see",
    "i wanna listen to", "i want to listen to", "i wanna hear",
    "i feel like watching", "i feel like listening to", "i feel like",
    "i wanna", "i want to", "i would like to", "id like to",
    "lets watch", "let's watch", "lets listen to", "let's listen to",
    "lets", "let's",
    # verbs
    "pull up", "bring up", "put on", "open up", "open", "launch", "go to",
    "search for", "search on youtube for", "search on youtube",
    "search youtube for", "search youtube", "search",
    "look up", "look for", "google",
    "find me", "find", "show me", "give me", "get me",
    "play me", "play", "watch", "see", "listen to", "hear",
    "throw on", "slap on", "queue up", "shuffle",
    "start", "just", "now",
    # determiners / quantifiers.
    # "the" is deliberately absent: it is far more often part of a title
    # ("the office", "the batman", "the weeknd") than filler, and stripping it
    # turned "put on the office" into a search for "office".
    "some more", "some", "a few", "a bit of", "any", "a", "an",
    "me", "us",
]

# Stripped from the end, same procedure.
SUFFIX_FILLER = [
    "right now", "right away", "real quick", "real fast", "quick", "quickly",
    "today", "tonight", "tomorrow", "now", "later",
    "please", "thanks", "thank you", "for me", "for us",
    "on youtube", "in youtube", "on yt", "from youtube",
    "on google", "in google",
    # Apple Music service phrasing. Stripped from the end so "play blinding
    # lights on apple music" leaves "blinding lights". Longer phrases first
    # (handled by the length sort) so "on apple music" beats "apple music".
    "on apple music", "in apple music", "on apple", "apple music", "apple",
    # Shuffle / order request shape, not part of any title.
    "on shuffle", "shuffled", "on normal", "in order", "no shuffle",
    "videos", "video", "clips", "clip",
    "songs", "song", "music", "track", "tracks", "album",
    # Trailing "radio" is a request shape ("travis scott radio"), not part of a
    # name. Interior "radio" is untouched, so "radio ga ga" survives.
    "radio", "on repeat", "playlist", "playlists",
    "stuff", "content", "something", "some",
    "and stuff", "or something",
    # Bare platform names left over from "i wanna watch youtube".
    "youtube", "google", "yt",
]

# Sorting by length means "i wanna watch" is tried before "i wanna", so the
# longer, more specific phrase wins and doesn't leave "watch" behind.
_PREFIXES = sorted(PREFIX_FILLER, key=len, reverse=True)
_SUFFIXES = sorted(SUFFIX_FILLER, key=len, reverse=True)


def _norm(text: str) -> str:
    """Lowercase, drop punctuation that isn't part of a term, collapse spaces."""
    text = (text or "").lower().strip()
    text = re.sub(r"[?!.,;:]+$", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _strip_end(text: str, phrases: list[str], leading: bool) -> tuple[str, bool]:
    """Remove one matching phrase from the given end. Returns (text, changed)."""
    for p in phrases:
        if leading:
            if text == p:
                return "", True
            if text.startswith(p + " "):
                return text[len(p) + 1:].strip(), True
        else:
            if text == p:
                return "", True
            if text.endswith(" " + p):
                return text[: -(len(p) + 1)].strip(), True
    return text, False


def extract(query: str, max_rounds: int = 12) -> str:
    """
    Strip filler from both ends and return what's left.

    Returns "" when the utterance is pure filler ("open youtube", "make it
    louder"), which is the correct answer for intents that take no argument.
    """
    text = _norm(query)

    for _ in range(max_rounds):
        changed = False
        text, c = _strip_end(text, _PREFIXES, leading=True)
        changed |= c
        text, c = _strip_end(text, _SUFFIXES, leading=False)
        changed |= c
        if not changed:
            break

    return text.strip()


# ─── Slot quality ────────────────────────────────────────────────────────────
# Stripping filler always returns *something*, and that something is not always
# an argument. "watch out" reduces to "out", which was enough to fire a YouTube
# search for the word "out". Two checks, deliberately different in strictness:
#
#   usable_slot()      — is this a legitimate argument at all?
#   looks_like_title() — is this specifically a piece of media to play?
#
# The second is stricter because it authorises an action with no scoring behind
# it, so its false positives are more expensive.

def _words(*terms: str) -> re.Pattern:
    return re.compile(r"\b(?:" + "|".join(terms) + r")\b", re.IGNORECASE)


# Residuals that are pure leftovers rather than arguments.
SLOT_STOPWORDS = {
    "out", "up", "down", "in", "on", "off", "it", "that", "this", "there",
    "here", "again", "now", "later", "one", "thing", "things", "stuff",
    "more", "back", "over", "too", "also", "still", "ok", "okay",
    # Articles and filler. Safe to list even though titles contain them,
    # because names_something() only needs one word from outside this set:
    # "the office" still keeps "office".
    "the", "and", "for", "with", "just", "please", "yeah", "some", "any",
    "really", "actually", "dude", "bro", "man",
}

# Second person means the sentence is about us, not about content:
# "show me what you're wearing" must never become a search.
#
# First-person object pronouns are deliberately absent. Song titles are full of
# them ("Next to Me", "Call Me Maybe", "You Belong With Me"), and blocking them
# made "play next to me by imagine dragons" ineligible for media routing, which
# then scored as a skip-track request. Chat lines stay protected because the
# leading "show me" / "tell me" is stripped before this runs, and because media
# routing additionally requires a play/watch verb or a media-type word.
_PRONOUNS = _words("you", "your", "youre", "yourself", "u", "i", "im")

# Additionally excluded from *media* routing. Question and feeling words appear
# in real searches ("how to tie a tie"), so they only block the rule-based
# action path, not ordinary slot use.
_NOT_MEDIA = _words(
    "what", "whats", "how", "why", "when", "where", "do", "does", "did",
    "are", "is", "am", "was", "can", "could", "would", "should",
    "tell", "think", "feel", "love", "miss",
)

MAX_TITLE_WORDS = 8

# Transport phrasings are about whatever is already playing, so they must never
# be routed as "play this entity": "play the next song" would start a new track
# instead of skipping. Declining only defers to the literal transport rules and
# to embedding scoring, both of which handle these correctly.
TRANSPORT_WORDS = _words(
    "pause", "unpause", "resume", "skip", "previous", "unmute",
    "stop", "halt", "kill", "hold",
)


def _tokens(slot: str) -> list[str]:
    return [w.strip(".,!?'\"") for w in slot.split() if w.strip(".,!?'\"")]


# Words that describe the *kind* of media rather than naming any of it. A slot
# built only from these names nothing: "start the music up again" reduces to
# "the music up again", which is a transport request, not a track to play.
GENERIC_MEDIA_WORDS = {
    "music", "song", "songs", "track", "tracks", "tune", "tunes", "playlist",
    "album", "video", "videos", "clip", "clips", "audio", "sound", "playback",
}


def names_something(slot: str) -> bool:
    """True when the slot contains a word that could actually be a title."""
    tokens = _tokens(slot)
    if not tokens:
        return False
    return any(
        t not in GENERIC_MEDIA_WORDS and t not in SLOT_STOPWORDS
        for t in tokens
    )


def usable_slot(slot: str) -> bool:
    """True when `slot` is a real argument rather than stripping residue."""
    if not slot or len(slot) < 2:
        return False
    tokens = _tokens(slot)
    if not tokens:
        return False
    # Judged across the whole slot: "this is america" keeps a content word,
    # while "it again" has none.
    if all(t in SLOT_STOPWORDS for t in tokens):
        return False
    if _PRONOUNS.search(slot):
        return False
    return True


def looks_like_title(slot: str) -> bool:
    """True when `slot` could plausibly name something to watch or play."""
    if not usable_slot(slot):
        return False
    if len(_tokens(slot)) > MAX_TITLE_WORDS:
        return False
    return not _NOT_MEDIA.search(slot)


# ─── Canonical rewriting ─────────────────────────────────────────────────────
# Each intent maps to a phrasing that commands.plan_command already matches, so
# a semantic hit reuses the existing handler rather than duplicating it.
#
# `None` means the intent takes no argument, and a slot found alongside it is
# discarded. `{}` is where the extracted slot lands.

CANONICAL = {
    "youtube_search": "search on youtube {}",
    "youtube_play": "play {} on youtube",
    "open_youtube": "open youtube",
    "open_google": "open google",
    "web_search": "search {}",
    "play_music": "play {} song",
    # Transport intents take no argument; a slot found alongside one is noise.
    "pause_music": "pause the music",
    "resume_music": "resume the music",
    "next_track": "next song",
    "previous_track": "previous song",
    "now_playing": "what song is this",
    "volume_up": "volume up",
    "volume_down": "volume down",
    "mute": "mute",
    "the_time": "the time",
    "wikipedia": "wikipedia {}",
    "joke": "joke",
    # Rewritten into phrasing the literal laughter rule recognises, so the
    # handler stays in commands.py like every other intent.
    "perform_laugh": "laugh for me",
    "speed_test": "speed test",
}

# Intents that are meaningless without an argument. Matching one with an empty
# slot should fall through to conversation rather than run a blank search.
NEEDS_SLOT = {"youtube_search", "youtube_play", "web_search", "wikipedia"}


def to_command(intent: str, query: str) -> str | None:
    """
    Rewrite a matched intent plus the original utterance into a canonical
    command string, or None if the intent needs an argument and none was found.
    """
    template = CANONICAL.get(intent)
    if template is None:
        return None

    if "{}" not in template:
        return template

    slot = extract(query)
    if intent in NEEDS_SLOT and not usable_slot(slot):
        # Covers both "no slot at all" and "slot is leftover filler". Running a
        # search for "out" is worse than treating the line as conversation.
        return None
    if not slot:
        return template.replace(" {}", "").replace("{}", "").strip()

    return template.format(slot)
