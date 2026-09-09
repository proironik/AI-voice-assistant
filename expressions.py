"""Voice-expression helpers for Chatterbox-Turbo.

The installed Turbo tokenizer has a fixed set of native bracket tokens. This
module keeps those control tokens out of the visible chat while preserving them
for speech, translates common LLM stage directions (``*laughs*``), and strips
invented tags before they can be spoken literally.
"""

import re
from dataclasses import dataclass

# Verified from the installed model's added_tokens.json. Do not add aliases here
# unless the tokenizer actually gains them; unknown bracket text is not a voice
# control and may be read aloud.
SUPPORTED_TAGS = frozenset({
    "angry",
    "fear",
    "surprised",
    "whispering",
    "advertisement",
    "dramatic",
    "narration",
    "crying",
    "happy",
    "sarcastic",
    "clear throat",
    "sigh",
    "shush",
    "cough",
    "groan",
    "sniff",
    "gasp",
    "chuckle",
    "laugh",
})

# Turbo can render all verified tags, but these are the natural conversational
# subset the LLM is encouraged to use. The rest remain accepted for explicit
# user-authored text.
CONVERSATIONAL_TAGS = (
    "chuckle",
    "laugh",
    "sigh",
    "gasp",
    "happy",
    "sarcastic",
    "whispering",
)

# One event per short 1-2 sentence reply keeps expression from becoming noisy.
MAX_TAGS_PER_REPLY = 1

_BRACKET_TAG = re.compile(r"\[\s*([^\[\]]+?)\s*\]", re.IGNORECASE)
_DANGLING_BRACKET = re.compile(r"\[[^\]]*$")
_STAGE_DIRECTION = re.compile(
    r"(?:\*|\()\s*"
    r"(laughs?|laughing|chuckles?|chuckling|giggles?|giggling|sighs?|sighing|"
    r"gasps?|gasping|whispers?|whispering)"
    r"\s*(?:\*|\))",
    re.IGNORECASE,
)
_LEADING_LAUGH = re.compile(
    r"^\s*(?:ha(?:ha)+|he(?:he)+|lol)\s*[,!.—-]*\s*",
    re.IGNORECASE,
)

# Stage directions with no matching voice token, such as *wink* or *smirks*.
# Turbo has no tag for them, so left in place they were simply pronounced: the
# assistant said the word "wink" out loud.
_UNMAPPED_DIRECTION = re.compile(r"\*\s*([^*\n]{1,40}?)\s*\*")

# Physical actions the voice cannot perform. A single asterisked word outside
# this set is treated as emphasis and kept, because deleting it changes the
# sentence: "*the* one thing I needed" should not become "one thing I needed".
_ACTION_WORDS = {
    "wink", "winks", "winking", "smirk", "smirks", "smirking",
    "blush", "blushes", "blushing", "grin", "grins", "grinning",
    "smile", "smiles", "smiling", "shrug", "shrugs", "shrugging",
    "pout", "pouts", "pouting", "moan", "moans", "moaning",
    "purr", "purrs", "purring", "bite", "bites", "biting",
    "lick", "licks", "licking", "kiss", "kisses", "kissing",
    "hug", "hugs", "hugging", "nod", "nods", "nodding",
    "stare", "stares", "staring", "shiver", "shivers", "shivering",
    "leans", "lean", "whimper", "whimpers", "teases", "tease",
}

# The LLM sometimes wraps its whole reply in quotes. The closing quote is not
# sentence-ending punctuation, so a period was appended after it and the UI
# showed lines finishing with ".
_WRAPPING_QUOTES = re.compile(r'^\s*["\u201c\u2018\']+(.*?)["\u201d\u2019\']+\s*$', re.DOTALL)

_STAGE_TO_TAG = {
    "laugh": "laugh",
    "laughs": "laugh",
    "laughing": "laugh",
    "chuckle": "chuckle",
    "chuckles": "chuckle",
    "chuckling": "chuckle",
    "giggle": "chuckle",
    "giggles": "chuckle",
    "giggling": "chuckle",
    "sigh": "sigh",
    "sighs": "sigh",
    "sighing": "sigh",
    "gasp": "gasp",
    "gasps": "gasp",
    "gasping": "gasp",
    "whisper": "whispering",
    "whispers": "whispering",
    "whispering": "whispering",
}


@dataclass(frozen=True)
class ExpressiveReply:
    """Visible prose plus the separate control-token-bearing speech text."""

    display: str
    speech: str
    tags: tuple[str, ...]


def _stage_replacement(match: re.Match) -> str:
    tag = _STAGE_TO_TAG.get(match.group(1).lower())
    return f"[{tag}]" if tag else ""


def _unmapped_replacement(match: re.Match) -> str:
    """Drop asterisked actions; keep asterisked emphasis as plain words."""
    inner = match.group(1).strip()
    words = inner.split()
    if len(words) > 1:
        # Multi-word asterisks are stage directions ("*bites lip*"), not
        # emphasis, which is almost always a single word.
        return ""
    return "" if words and words[0].lower() in _ACTION_WORDS else inner


def _clean_spacing(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    # Removing a tag/stage direction can leave punctuation at the front.
    text = re.sub(r"^[,;:]\s*", "", text)
    return text.strip()


def prepare(text: str, max_tags: int = MAX_TAGS_PER_REPLY) -> ExpressiveReply:
    """Return display text and safe Chatterbox speech text for an LLM reply.

    Exact supported tags are kept up to ``max_tags``. Unknown tags are removed,
    repeated/excess tags are removed, and common textual stage directions are
    mapped to native tokens. A leading textual ``haha``/``hehe`` becomes a
    native chuckle rather than being pronounced as syllables.

    The spoken text always covers the whole reply. Speaking only a leading
    clause was tried and dropped: it made the voice contradict the visible
    text, so reply length is now controlled at the LLM instead.
    """
    text = text or ""
    # Unwrap before anything else, so trailing punctuation logic sees the real
    # final character rather than a quote mark.
    quoted = _WRAPPING_QUOTES.match(text)
    if quoted and '"' not in quoted.group(1):
        text = quoted.group(1)

    text = _STAGE_DIRECTION.sub(_stage_replacement, text)
    # Anything still in asterisks has no voice token behind it.
    text = _UNMAPPED_DIRECTION.sub(_unmapped_replacement, text)
    # Token-capped LLM output can end halfway through an invented cue such as
    # "[wink". It is neither display prose nor a valid tokenizer control.
    text = _DANGLING_BRACKET.sub("", text)

    if _LEADING_LAUGH.search(text) and not _BRACKET_TAG.search(text):
        text = _LEADING_LAUGH.sub("[chuckle] ", text, count=1)

    accepted: list[str] = []

    def keep_supported(match: re.Match) -> str:
        tag = match.group(1).strip().lower()
        if tag not in SUPPORTED_TAGS:
            return ""
        if len(accepted) >= max_tags or tag in accepted:
            return ""
        accepted.append(tag)
        return f"[{tag}]"

    speech = _clean_spacing(_BRACKET_TAG.sub(keep_supported, text))
    display = _clean_spacing(_BRACKET_TAG.sub("", speech))

    # A tight token cap can end on a complete clause without punctuation. Keep
    # both visible and spoken forms natural rather than displaying a cut-off
    # looking line or relying on Chatterbox alone to add the final period.
    if display and display[-1] not in ".!?":
        display += "."
        speech += "."

    # A reply that consists only of a vocal event is valid speech, but the text
    # box should still communicate what happened.
    if not display and accepted:
        display = f"({accepted[0]}s)" if accepted[0] not in {"happy", "sarcastic"} else "…"

    return ExpressiveReply(display=display, speech=speech, tags=tuple(accepted))


def strip_tags(text: str) -> str:
    """Remove every bracket control token for logs or visible UI text."""
    return _clean_spacing(_BRACKET_TAG.sub("", text or ""))
