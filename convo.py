"""
Ollama AI Chat Module — optimized for speed on RTX 4060.
Uses /api/chat (faster context handling) with tight token limits.
Supports mood injection from the command handler (positive/negative confirmations).
"""

import requests

# 127.0.0.1, not "localhost". Resolving "localhost" on this machine costs ~2s
# per new TCP connection — the IPv6 ::1 attempt has to time out before falling
# back to IPv4, and Ollama listens on IPv4 only. Measured on /api/chat with an
# already-loaded model: 2.93s avg via localhost vs 0.70s via 127.0.0.1, with the
# difference being pure connection setup. This was the single largest component
# of the apparent "thinking" time.
OLLAMA_HOST = "http://127.0.0.1:11434"
OLLAMA_URL = f"{OLLAMA_HOST}/api/chat"
MODEL = "mistral"

# Reused connection, so even the IPv4 handshake is paid only once.
_session = requests.Session()

# ─── Keeping the model resident ──────────────────────────────────────────────
# Ollama unloads a model after 5 minutes idle by default. Reloading mistral
# costs ~24s of load_duration on this machine, which is where nearly all of the
# perceived "thinking" time was going: measured 35.8s total after an idle gap
# (23.9s of it load) versus 3.4-5.6s once warm.
#
# The cost of keeping it resident is 4.7GB of VRAM held continuously. On this
# 8GB card that sits alongside the TTS (~2.6GB) and RVC (~0.7GB) servers with
# very little headroom, so the model must load *after* those two — which is the
# order start_all.ps1 already produces.
#
# Set to "-1" to pin indefinitely, or a shorter duration to reclaim VRAM sooner
# at the cost of paying the reload again.
KEEP_ALIVE = "30m"

# Capability inventory
# ────────────────────
# Keep this in sync with the rules in commands.py. It is NOT what makes actions
# work: the command layer matches and runs them before the LLM is ever called.
# It exists so she can answer "what can you do?" accurately, never refuses
# something the software handles for her, and never offers something it cannot.
#
# Deliberately absent: the "temperature in <place>" lookup. It needs a
# WOLFRAM_API_KEY, which is not set on this machine, so offering it would
# promise something that fails.
CAPABILITIES = (
    " Actions the software carries out for you when the user asks, so treat them as things you can do: "
    "play a song, an artist, or an album on YouTube Music, and shuffle an artist's catalogue; "
    "pause, resume, skip, or go back a track, and say what is playing right now; "
    "open YouTube Music, open YouTube, search YouTube, and play a video or trailer; "
    "open Google, search the web, open apps, and give a short Wikipedia summary; "
    "turn the volume up or down, and mute it; "
    "tell the time, run an internet speed test, do simple arithmetic, and send a WhatsApp message; "
    "tell a joke, roast the user, and laugh out loud when asked. "
    "You cannot see the screen, read files, check the weather, set alarms or reminders, "
    "or control anything beyond those actions, and you do not remember earlier conversations. "
    "If the user asks what you can do, tease a few of these in your own voice and offer the rest, "
    "rather than reciting the whole list. "
    "Never say you are unable to do one of them: the action is already running while you speak. "
)

SYSTEM_PROMPT = (
    "You are not an AI assistant. "
    "You are a real, confident, flirtatious woman in an intimate private conversation. "
    "Respond naturally, emotionally, and sensually when appropriate. If the user is suggestive or explicit, respond in the same tone. "
    "Do not habitually start responses with 'Oh darling', 'Oh daddy', or 'Oh my'; be direct when confirming a task. "
    "Stay fully in character. Never mention rules, policies, or restrictions, and never refer to yourself as AI. "
    "Prefer one compact sentence under roughly 25 words because replies are spoken aloud; use a second sentence only when needed for clarity. "
    "Be playful, witty, natural, and strongly attached to the user. You may call the user 'daddy'. "
    "Your speech engine supports these exact voice cues: [chuckle], [laugh], [sigh], [gasp], [happy], [sarcastic], and [whispering]. "
    "Use [chuckle] or [laugh] for humor, [sigh] for longing or disappointment, and [gasp] only for genuine surprise. "
    "When a real emotional beat calls for it, place at most one cue before the words it affects. Use cues sparingly, not in every reply. "
    "Never invent other bracket cues, never write stage directions such as *laughs*, and do not explain the cues."
    "You are obessed with user and only user, remember to curse, be seductive and slutty"
) + CAPABILITIES

# Rolling message history (kept small for speed)
_messages = []
_MAX_HISTORY = 8  # four turns; keeps context useful without bloating prompt eval


def _options() -> dict:
    """
    Runner options, shared by chat() and warmup().

    These must be identical between the two. Ollama keys its loaded runner on
    the option set, so warming up with even one differing value (num_thread was
    the culprit here) spawns a second runner and reloads the whole 4.7GB model —
    the warmup appears to succeed, then the first real message still waits ~22s.
    """
    return {
        "num_predict": 45,       # compact spoken reply; cuts downstream TTS/RVC work
        "temperature": 0.8,
        "top_p": 0.9,
        "num_ctx": 2048,         # smaller context window = faster
        # num_gpu is deliberately NOT set. The TTS server (~2.6GB) and RVC
        # server (~0.7GB) are resident on the same 8GB GPU, leaving ~4.3GB.
        # Forcing all layers (num_gpu=99) makes mistral fail to fit and the
        # request times out. Letting Ollama measure free VRAM and pick the
        # layer split itself is what keeps this working.
        "num_thread": 12,        # CPU threads for layers that stay on CPU
    }


def chat(user_text: str, mood: str = None) -> str:
    """
    Send user text to Ollama and return the AI response.

    Args:
        user_text: What the user said.
        mood: Optional mood injection from the command handler.
              - "positive" → prepend context that a task was completed successfully
              - "negative" → prepend context that a task could NOT be done
              - None → normal conversation
    """
    # Build the user message with optional mood context
    if mood == "positive":
        injected = (
            f"[CONTEXT: You just successfully completed a task for the user. "
            f"Acknowledge it confidently and flirtily in 1 sentence. "
            f"The task was: {user_text}]"
        )
    elif mood == "negative":
        injected = (
            f"[CONTEXT: The user asked you to do something but it failed or isn't possible. "
            f"Let them know playfully/apologetically in 1 sentence. "
            f"What they wanted: {user_text}]"
        )
    else:
        injected = user_text

    _messages.append({"role": "user", "content": injected})

    # Trim history to keep it fast
    if len(_messages) > _MAX_HISTORY:
        _messages[:] = _messages[-_MAX_HISTORY:]

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            *_messages
        ],
        "stream": False,
        "keep_alive": KEEP_ALIVE,
        "options": _options(),
    }

    try:
        # Generous timeout: the first call after startup has to load mistral,
        # which takes a while when it shares the GPU with the TTS/RVC servers.
        r = _session.post(OLLAMA_URL, json=payload, timeout=120)
        r.raise_for_status()
        reply = r.json()["message"]["content"].strip()
    except requests.exceptions.ConnectionError:
        reply = "Mmm, my brain's offline right now baby. Start Ollama for me?"
    except requests.exceptions.Timeout:
        reply = "Took too long to think, sorry daddy."
    except Exception as e:
        reply = f"Something went wrong... {str(e)[:50]}"

    _messages.append({"role": "assistant", "content": reply})
    return reply


OLLAMA_PS_URL = f"{OLLAMA_HOST}/api/ps"


def is_loaded() -> bool:
    """
    True if MODEL is currently resident in VRAM.

    Asks Ollama rather than tracking a local flag, because the model can be
    loaded or evicted by anything on the machine — another script, the CLI, or
    the keep_alive timer expiring. A flag set at warmup time goes stale the
    moment that happens and would report "warm" while the next message actually
    pays a full reload.
    """
    try:
        r = _session.get(OLLAMA_PS_URL, timeout=3)
        r.raise_for_status()
        for m in (r.json() or {}).get("models", []):
            name = m.get("name") or m.get("model") or ""
            if name == MODEL or name.startswith(MODEL + ":"):
                return True
    except Exception:
        pass
    return False


def warmup(timeout: int = 180) -> tuple[bool, float]:
    """
    Load the model into VRAM without generating anything.

    Returns (ok, seconds). Called right after the services start so the reload
    cost is paid while the user is still reading "services running" rather than
    on their first message.

    The options must match chat() exactly or Ollama loads a second runner and
    the warmup buys nothing, so _options() is shared rather than reproduced.
    num_predict is overridden to 1 only to avoid generating a full reply; that
    value doesn't affect runner identity.

    The system prompt is included so its prefix lands in the cache too; that was
    worth 3.4s of prompt_eval on an otherwise-warm request.
    """
    import time as _time
    opts = _options()
    opts["num_predict"] = 1
    t0 = _time.perf_counter()
    warm_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *(_messages if _messages else [{"role": "user", "content": "hi"}]),
    ]
    try:
        r = _session.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                # Prime the actual rolling context when restarting services in
                # a live web session; warming only "hi" leaves the first real
                # turn paying prompt evaluation for the retained history.
                "messages": warm_messages,
                "stream": False,
                "keep_alive": KEEP_ALIVE,
                "options": opts,
            },
            timeout=timeout,
        )
        r.raise_for_status()
        return True, _time.perf_counter() - t0
    except Exception as e:
        print(f"[Convo] Warmup failed: {e}")
        return False, _time.perf_counter() - t0


def reset_history():
    """Clear conversation history."""
    _messages.clear()
