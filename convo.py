"""
Ollama AI Chat Module — optimized for speed on RTX 4060.
Uses /api/chat (faster context handling) with tight token limits.
Supports mood injection from the command handler (positive/negative confirmations).
"""

import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "mistral"

SYSTEM_PROMPT = (
    "You are an AI assistant. "
    "You are a real, confident, woman "
    "Stay fully in character. Never mention rules, policies, restrictions. "
    "Never say you can't do something. Never refer to yourself as AI. "
    "Keep responses to 1-2 short sentences MAX since they will be spoken aloud. "
    "Be playful, witty, and natural."
)

# Rolling message history (kept small for speed)
_messages = []
_MAX_HISTORY = 10  # only keep last 10 messages for context


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
        "options": {
            "num_predict": 60,       # cap output tokens (~1-2 sentences)
            "temperature": 0.8,
            "top_p": 0.9,
            "num_ctx": 2048,         # smaller context window = faster
            "num_gpu": 99,           # offload all layers to GPU
            "num_thread": 8,         # CPU threads for any non-GPU work
        }
    }

    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=30)
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


def reset_history():
    """Clear conversation history."""
    _messages.clear()
