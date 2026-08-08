"""
RVC Client Module
─────────────────
Connects to the persistent RVC server (rvc_server.py) over TCP.
If the server isn't running, falls back to subprocess (slow).

The server approach eliminates model reload time (~10-15s) on every call.
"""

import json
import socket
import subprocess
from pathlib import Path
import os

# ─── Server connection settings ──────────────────────────────────────────────

RVC_HOST = "127.0.0.1"
RVC_PORT = 7865
SOCKET_TIMEOUT = 60  # max wait for conversion

# ─── Subprocess fallback settings ────────────────────────────────────────────

RVC_ROOT = Path(r"D:\rvc\Retrieval-based-Voice-Conversion-WebUI")
RVC_PYTHON = Path(r"D:\rvc\rvc_env\Scripts\python.exe")
INFER_CLI = "tools/infer_cli.py"
MODEL_NAME = "Mom.pth"
INDEX_PATH = RVC_ROOT / "assets" / "indices" / "added_IVF1124_Flat_nprobe_1_Mom_v2.index"


def _try_server(input_wav: str, output_wav: str) -> bool:
    """
    Try to use the persistent RVC server.
    Returns True if successful, False if server unreachable.
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(SOCKET_TIMEOUT)
        sock.connect((RVC_HOST, RVC_PORT))

        request = json.dumps({
            "input": str(Path(input_wav).resolve()),
            "output": str(Path(output_wav).resolve()),
        })
        sock.sendall(request.encode("utf-8"))
        sock.shutdown(socket.SHUT_WR)  # signal end of request

        # Read response
        data = b""
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            data += chunk

        sock.close()

        response = json.loads(data.decode("utf-8"))
        if response["status"] == "ok":
            return True
        else:
            print(f"[RVC Client] Server error: {response.get('message')}")
            return False

    except (ConnectionRefusedError, socket.timeout, OSError):
        return False


def _subprocess_fallback(input_wav: str, output_wav: str):
    """Cold-start subprocess fallback (slow, ~15-20s)."""
    print("[RVC Client] Server unavailable, using subprocess fallback (slow)...")
    env = os.environ.copy()
    env["PATH"] = r"D:\rvc\rvc_env\Scripts;" + env["PATH"]

    cmd = [
        str(RVC_PYTHON),
        INFER_CLI,
        "--model_name", MODEL_NAME,
        "--input_path", str(Path(input_wav).resolve()),
        "--opt_path", str(Path(output_wav).resolve()),
        "--f0method", "rmvpe",
        "--index_path", str(INDEX_PATH),
        "--index_rate", "0.8",
    ]

    subprocess.run(cmd, cwd=str(RVC_ROOT), check=True, env=env)


def run_rvc(input_wav: str, output_wav: str) -> str:
    """
    Convert voice using RVC.
    Tries persistent server first (fast), falls back to subprocess (slow).
    """
    if not _try_server(input_wav, output_wav):
        _subprocess_fallback(input_wav, output_wav)

    return output_wav
