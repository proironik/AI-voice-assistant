"""
Service Manager
───────────────
Detects, starts, and safely stops the three backend services the assistant
depends on:

    ollama  : LLM            (HTTP  127.0.0.1:11434)
    tts     : Chatterbox TTS (TCP   127.0.0.1:7866)
    rvc     : RVC conversion (TCP   127.0.0.1:7865)

Liveness is decided by whether the port accepts a connection, not by whether a
process exists. A process that is still loading its model hasn't bound its port
yet, so port-based checks are what actually tell you the service is usable.

Shutdown is graceful-first: taskkill without /F asks the process to close and
lets it run its own cleanup (release VRAM, close sockets). Only if it is still
alive after GRACE_PERIOD do we escalate to /F. That ordering matters most for
Ollama, which maintains its own model cache on disk.
"""

import json
import os
import socket
import subprocess
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
START_SCRIPT = os.path.join(ROOT, "start_all.ps1")

# name -> (host, port, human label)
SERVICES = {
    "ollama": ("127.0.0.1", 11434, "Ollama"),
    "tts": ("127.0.0.1", 7866, "Chatterbox TTS"),
    "rvc": ("127.0.0.1", 7865, "RVC"),
}

PORT_TIMEOUT = 0.35     # seconds; local ports answer fast or not at all
GRACE_PERIOD = 8.0      # seconds to wait for a polite shutdown
POLL_INTERVAL = 0.4

# Scripts we own, matched against process command lines.
OWNED_SCRIPTS = ("tts_server.py", "rvc_server.py")
OLLAMA_IMAGES = ("ollama.exe", "ollama app.exe")


# ─── Status ──────────────────────────────────────────────────────────────────

def _port_open(host: str, port: int, timeout: float = PORT_TIMEOUT) -> bool:
    """True if something is listening on host:port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        return s.connect_ex((host, port)) == 0


def is_up(name: str) -> bool:
    """Check one service without probing every unrelated backend."""
    try:
        host, port, _ = SERVICES[name]
    except KeyError as e:
        raise ValueError(f"Unknown service: {name}") from e
    return _port_open(host, port)


def status() -> dict:
    """
    Report which services are reachable.

    Returns:
        {
          "services": {"ollama": True, "tts": False, "rvc": True},
          "labels":   {"ollama": "Ollama", ...},
          "all_up": False,
          "any_up": True,
        }
    """
    up = {
        name: _port_open(host, port)
        for name, (host, port, _) in SERVICES.items()
    }
    return {
        "services": up,
        "labels": {name: meta[2] for name, meta in SERVICES.items()},
        "all_up": all(up.values()),
        "any_up": any(up.values()),
    }


def wait_until_up(names=None, timeout: float = 90.0) -> dict:
    """Poll until the named services are reachable, or timeout elapses."""
    names = names or list(SERVICES)
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = status()
        if all(st["services"][n] for n in names):
            return st
        time.sleep(POLL_INTERVAL)
    return status()


# ─── Process discovery ───────────────────────────────────────────────────────

def _run(args, timeout=20):
    """Run a command, returning stdout as text. Never raises on non-zero exit."""
    try:
        p = subprocess.run(
            args,
            capture_output=True,
            timeout=timeout,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return p.stdout.decode("utf-8", errors="replace")
    except Exception:
        return ""


def _powershell(script: str, timeout=25) -> str:
    return _run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
        timeout=timeout,
    )


def find_pids() -> dict:
    """
    Map service name -> list of PIDs.

    Our two servers are plain `python <script>.py` invocations, so they are
    identified by command line rather than image name; several unrelated python
    processes (including this web server) are typically also running.
    """
    pids = {"ollama": [], "tts": [], "rvc": []}

    out = _powershell(
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.Name -like 'python*' -or $_.Name -like 'ollama*' } | "
        "ForEach-Object { \"$($_.ProcessId)`t$($_.Name)`t$($_.CommandLine)\" }"
    )

    for line in out.splitlines():
        parts = line.split("\t", 2)
        if len(parts) < 2:
            continue
        raw_pid, image = parts[0].strip(), parts[1].strip()
        cmdline = parts[2] if len(parts) > 2 else ""
        if not raw_pid.isdigit():
            continue
        pid = int(raw_pid)

        if image.lower() in (i.lower() for i in OLLAMA_IMAGES):
            pids["ollama"].append(pid)
        elif "tts_server.py" in cmdline:
            pids["tts"].append(pid)
        elif "rvc_server.py" in cmdline:
            pids["rvc"].append(pid)

    return pids


def _alive(pid: int) -> bool:
    out = _run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], timeout=10)
    return str(pid) in out


# ─── Start ───────────────────────────────────────────────────────────────────

def start_all() -> dict:
    """
    Launch anything that isn't already running, via start_all.ps1.

    Returns a dict describing what was skipped (already up) and what was
    launched. The script itself opens each server in its own window, so this
    returns as soon as the launcher exits, not when the models finish loading.
    """
    before = status()
    already = [n for n, ok in before["services"].items() if ok]

    if before["all_up"]:
        # Same keys as the launch path below. Callers branch on "services", and
        # omitting it here made them treat an all-up result as though nothing
        # was running.
        return {
            "status": "ok",
            "launched": [],
            "already_running": already,
            "failed": [],
            "services": before["services"],
            "message": "All services were already running.",
        }

    try:
        subprocess.Popen(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
             "-File", START_SCRIPT],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        return {"status": "error", "message": f"Could not run start_all.ps1: {e}"}

    to_wait = [n for n, ok in before["services"].items() if not ok]
    after = wait_until_up(to_wait, timeout=90.0)

    launched = [n for n in to_wait if after["services"][n]]
    failed = [n for n in to_wait if not after["services"][n]]

    return {
        "status": "ok" if not failed else "partial",
        "launched": launched,
        "already_running": already,
        "failed": failed,
        "services": after["services"],
        "message": (
            "All services are up."
            if not failed
            else "Timed out waiting for: "
                 + ", ".join(SERVICES[n][2] for n in failed)
        ),
    }


# ─── Safe stop ───────────────────────────────────────────────────────────────

def _request_socket_shutdown(name: str) -> bool:
    """
    Ask tts_server.py / rvc_server.py to shut themselves down via their own
    protocol: {"command": "shutdown"}.

    This is the safe path. The server finishes any in-flight work, closes its
    listener, frees CUDA cache, and exits on its own. Returns True if the port
    stopped accepting connections within GRACE_PERIOD.
    """
    host, port, _ = SERVICES[name]
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(GRACE_PERIOD)
            s.connect((host, port))
            s.sendall(json.dumps({"command": "shutdown"}).encode("utf-8"))
            s.shutdown(socket.SHUT_WR)
            # Response is advisory; the port check below is what we trust.
            try:
                s.recv(1024)
            except OSError:
                pass
    except OSError:
        return False

    deadline = time.time() + GRACE_PERIOD
    while time.time() < deadline:
        if not _port_open(host, port):
            return True
        time.sleep(POLL_INTERVAL)
    return False


def _stop_pid(pid: int, grace: float = GRACE_PERIOD) -> str:
    """
    Ask a process to close, then force only if it refuses.

    Returns "graceful", "forced", "already-gone", or "failed".
    """
    if not _alive(pid):
        return "already-gone"

    # No /F: Windows posts a close request the process can act on.
    _run(["taskkill", "/PID", str(pid)], timeout=10)

    deadline = time.time() + grace
    while time.time() < deadline:
        if not _alive(pid):
            return "graceful"
        time.sleep(POLL_INTERVAL)

    _run(["taskkill", "/F", "/PID", str(pid)], timeout=10)
    time.sleep(0.5)
    return "failed" if _alive(pid) else "forced"


def _stop_ollama(pids) -> list:
    """
    Stop Ollama.

    `ollama stop` first: that unloads the running model through Ollama's own
    API, which is the only step where an abrupt kill could interrupt real work
    (it maintains a model store on disk). The processes are then closed, with
    "ollama app.exe" — the tray parent — taken last so it doesn't respawn the
    serve child it supervises.
    """
    results = []

    listed = _run(["ollama", "ps"], timeout=15)
    for line in listed.splitlines()[1:]:
        model = line.split()[0] if line.split() else None
        if model:
            _run(["ollama", "stop", model], timeout=30)
            results.append({"model": model, "result": "unloaded"})

    # Child serve processes before the tray app that owns them.
    ordered = sorted(pids, key=lambda p: _is_ollama_app(p))
    for pid in ordered:
        results.append({"pid": pid, "result": _stop_pid(pid)})

    return results


def _is_ollama_app(pid: int) -> bool:
    """True for 'ollama app.exe' (the tray parent), False for 'ollama.exe'."""
    out = _run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], timeout=10)
    return "ollama app.exe" in out.lower()


def stop_all() -> dict:
    """
    Stop TTS, RVC, then Ollama, preferring each service's own shutdown path.

    Order matters: TTS and RVC hold the bulk of the VRAM, and Ollama sizes its
    layer split against whatever is free, so the big consumers go first.

    For TTS and RVC the in-protocol shutdown is tried before any taskkill, so a
    conversion in progress is allowed to finish writing its WAV rather than
    being cut off mid-write.
    """
    pids = find_pids()
    details = {}

    for name in ("tts", "rvc"):
        if _request_socket_shutdown(name):
            details[name] = [{"result": "graceful (protocol shutdown)"}]
            # The venv stub parent can linger after the child exits.
            leftovers = [p for p in pids[name] if _alive(p)]
            for pid in leftovers:
                details[name].append({"pid": pid, "result": _stop_pid(pid, grace=3.0)})
        elif pids[name]:
            details[name] = [{"pid": pid, "result": _stop_pid(pid)} for pid in pids[name]]
        else:
            details[name] = [{"result": "not running"}]

    details["ollama"] = _stop_ollama(pids["ollama"]) if pids["ollama"] else [
        {"result": "not running"}
    ]

    after = status()
    still_up = [SERVICES[n][2] for n, ok in after["services"].items() if ok]

    return {
        "status": "ok" if not still_up else "partial",
        "details": details,
        "services": after["services"],
        "message": (
            "All services stopped."
            if not still_up
            else "Still reachable: " + ", ".join(still_up)
        ),
    }
