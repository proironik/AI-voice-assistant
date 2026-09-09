"""
Application Control (safe mode)
───────────────────────────────
Open apps by name, and close user apps gracefully. No force-kill: closing asks
the app's windows to close, so unsaved-work prompts still appear, exactly as if
the user clicked the X.

Two safety rails on closing
───────────────────────────
1. A hard denylist of processes that are never touched: OS-critical processes
   (explorer, winlogon, csrss, services, svchost, ...) and the assistant's own
   stack (its Python, Ollama, the TTS/RVC servers). Killing any of those would
   break the desktop or the assistant mid-command.

2. Graceful termination only. On Windows, psutil's terminate() posts WM_CLOSE
   to GUI processes, so the app can prompt to save. A process that ignores it
   is left running rather than escalated to a kill — the user is told it would
   not close on its own.

Opening
───────
Resolution order: an alias map for common apps first (so "chrome" reliably maps
to chrome.exe rather than whatever the shell guesses), then the Windows shell
resolver via `start`, which handles PATH executables, registered App Paths, and
Store apps. If neither resolves, the launch reports failure so the assistant
can say so instead of pretending.
"""

import os
import subprocess
import time

# ─── Opening ─────────────────────────────────────────────────────────────────

# Friendly name -> what to hand the Windows shell. Values are either an
# executable the shell can find on PATH/App Paths, a shell: folder, or a Store
# app execution alias. Keys are matched case-insensitively after stripping.
_ALIASES = {
    "chrome": "chrome",
    "google chrome": "chrome",
    "edge": "msedge",
    "microsoft edge": "msedge",
    "firefox": "firefox",
    "brave": "brave",
    "opera": "opera",
    "opera gx": "opera",
    "notepad": "notepad",
    "notepad++": "notepad++",
    "wordpad": "write",
    "word": "winword",
    "microsoft word": "winword",
    "excel": "excel",
    "powerpoint": "powerpnt",
    "outlook": "outlook",
    "calculator": "calc",
    "calc": "calc",
    "paint": "mspaint",
    "ms paint": "mspaint",
    "file explorer": "explorer",
    "explorer": "explorer",
    "files": "explorer",
    "task manager": "taskmgr",
    "control panel": "control",
    "settings": "ms-settings:",
    "cmd": "cmd",
    "command prompt": "cmd",
    "powershell": "powershell",
    "terminal": "wt",
    "windows terminal": "wt",
    "spotify": "spotify",
    "discord": "discord",
    "steam": "steam",
    "vlc": "vlc",
    "vs code": "code",
    "vscode": "code",
    "visual studio code": "code",
    "code": "code",
    "obs": "obs",
    "photoshop": "photoshop",
    "apple music": "music:",
    "music": "music:",
    "camera": "microsoft.windows.camera:",
    "photos": "ms-photos:",
    "store": "ms-windows-store:",
    "microsoft store": "ms-windows-store:",
    "snipping tool": "ms-screenclip:",
}


def resolve_target(name: str) -> str | None:
    """Map a spoken app name to a shell target, or None if we have no idea."""
    key = (name or "").strip().lower()
    if not key:
        return None
    if key in _ALIASES:
        return _ALIASES[key]
    # Not a known alias. Hand the raw name to the shell resolver and let Windows
    # try — this catches apps on PATH or with registered App Paths.
    return key


def open_app(name: str) -> bool:
    """
    Launch an app by name. Returns True if the launch was accepted.

    Uses `cmd /c start`, the same resolver the Run dialog uses, so it finds
    executables, App Paths, and Store aliases without a hardcoded path. The
    empty "" title argument is required by start when the target is quoted.
    """
    target = resolve_target(name)
    if not target:
        return False
    try:
        # start returns immediately; a nonzero exit means it could not resolve
        # the target. shell=True so cmd's start builtin is available.
        result = subprocess.run(
            f'start "" "{target}"',
            shell=True,
            timeout=10,
            capture_output=True,
        )
        ok = result.returncode == 0
        print(f"[AppControl] open {name!r} -> {target!r} "
              f"({'ok' if ok else 'failed rc=%d' % result.returncode})")
        return ok
    except Exception as e:
        print(f"[AppControl] open {name!r} failed: {e}")
        return False


# ─── Closing ─────────────────────────────────────────────────────────────────

# Never terminated. Lowercased process names (with and without .exe are both
# checked). OS-critical first, then the assistant's own runtime.
_PROTECTED = {
    # Windows core — killing these breaks the session or forces a reboot.
    "system", "system idle process", "registry", "smss", "csrss", "wininit",
    "winlogon", "services", "lsass", "svchost", "explorer", "dwm",
    "fontdrvhost", "ctfmon", "spoolsv", "sihost", "taskhostw", "runtimebroker",
    "shellexperiencehost", "startmenuexperiencehost", "searchhost",
    "audiodg", "conhost", "dllhost",
    # The assistant's own stack — closing these would kill the thing running
    # the command, or its voice/LLM backends.
    "python", "pythonw", "ollama", "ollama app",
}

# Substrings that also mark a process as the assistant's own backend, matched
# against the full command line so we catch "python web_server.py",
# "tts_server.py", "rvc_server.py" regardless of how they were launched.
_PROTECTED_CMDLINE = (
    "web_server.py", "tts_server.py", "rvc_server.py", "convo.py", "main.py",
)


def _is_protected(proc) -> bool:
    """True when a process must never be closed."""
    try:
        pname = (proc.name() or "").lower()
    except Exception:
        return True  # if we cannot even read it, do not touch it
    base = pname[:-4] if pname.endswith(".exe") else pname
    if base in _PROTECTED or pname in _PROTECTED:
        return True
    try:
        cmdline = " ".join(proc.cmdline()).lower()
        if any(hint in cmdline for hint in _PROTECTED_CMDLINE):
            return True
    except Exception:
        # No cmdline access (permissions) — be conservative and skip it.
        pass
    return False


def _matches(proc, needle: str) -> bool:
    """True when a process looks like the app the user named."""
    needle = needle.lower().strip()
    try:
        pname = (proc.name() or "").lower()
    except Exception:
        return False
    base = pname[:-4] if pname.endswith(".exe") else pname
    # alias -> executable, so "chrome" matches chrome.exe even though the spoken
    # word is not the process name.
    alias_target = _ALIASES.get(needle, needle)
    alias_base = alias_target[:-4] if alias_target.endswith(".exe") else alias_target
    return needle in base or base == alias_base or alias_base in base


def close_status(name: str) -> str:
    """
    Read-only pre-flight for a close request. Returns one of:

        "closable"     at least one non-protected process matches
        "protected"    matches only processes we refuse to touch
        "not_running"  nothing matches
        "unavailable"  psutil missing

    Used before the confirmation is spoken, so the assistant does not say
    "closing X" for something protected or not even running.
    """
    name = (name or "").strip()
    if not name:
        return "not_running"
    try:
        import psutil
    except Exception:
        return "unavailable"

    protected_hit = False
    for proc in psutil.process_iter():
        if not _matches(proc, name):
            continue
        if _is_protected(proc):
            protected_hit = True
            continue
        return "closable"
    return "protected" if protected_hit else "not_running"


def close_app(name: str, settle: float = 2.0) -> tuple[bool, str]:
    """
    Gracefully close every matching, non-protected process.

    Returns (closed_any, reason). `reason` is empty on success, or a short
    explanation the assistant can speak on failure.
    """
    name = (name or "").strip()
    if not name:
        return False, "no app named"

    try:
        import psutil
    except Exception as e:
        return False, f"process control unavailable ({e})"

    targets = []
    protected_hit = False
    for proc in psutil.process_iter():
        if not _matches(proc, name):
            continue
        if _is_protected(proc):
            protected_hit = True
            continue
        targets.append(proc)

    if not targets:
        if protected_hit:
            # Matched only something we refuse to touch (e.g. "close python").
            return False, "that's a protected system or assistant process"
        return False, "it doesn't look like that's running"

    # Ask each window to close. terminate() posts WM_CLOSE on Windows, so save
    # prompts appear; we do not escalate to kill().
    for proc in targets:
        try:
            proc.terminate()
        except Exception as e:
            print(f"[AppControl] terminate {proc} failed: {e}")

    gone, alive = psutil.wait_procs(targets, timeout=settle)
    if alive:
        # Some windows are still up — likely a "save changes?" dialog. Leave
        # them; forcing would lose data, which safe mode explicitly avoids.
        print(f"[AppControl] {len(alive)} process(es) still open after close "
              f"request for {name!r}")
        return (len(gone) > 0), (
            "it's asking me to save or won't close on its own"
            if not gone else ""
        )

    print(f"[AppControl] closed {len(gone)} process(es) for {name!r}")
    return True, ""
