"""
RVC Persistent Server
─────────────────────
Loads the RVC model ONCE and stays running, accepting conversion
requests over a local TCP socket. This eliminates the ~10-15s cold
start on every call.

Run this BEFORE launching main.py:
    python rvc_server.py

Protocol (JSON over TCP):
    Request:  {"input": "C:/path/to/tts.wav", "output": "C:/path/to/final.wav"}
    Response: {"status": "ok", "output": "C:/path/to/final.wav"}
              {"status": "error", "message": "..."}
"""

import json
import socket
import sys
import os
import threading
from pathlib import Path

# ─── RVC Setup ───────────────────────────────────────────────────────────────

RVC_ROOT = Path(r"D:\rvc\Retrieval-based-Voice-Conversion-WebUI")
sys.path.insert(0, str(RVC_ROOT))

os.chdir(str(RVC_ROOT))
os.environ["PATH"] = r"D:\rvc\rvc_env\Scripts;" + os.environ["PATH"]

MODEL_NAME = "Mom.pth"
INDEX_PATH = str(RVC_ROOT / "assets" / "indices" / "added_IVF1124_Flat_nprobe_1_Mom_v2.index")
F0_METHOD = "rmvpe"
INDEX_RATE = 0.8

# Import RVC internals after path setup
# These imports depend on your RVC installation — adjust if needed
try:
    # RVC reads weight_root / index_root / rmvpe_root from .env — without this
    # load_dotenv() the paths resolve to None and get_vc() fails on "None/Mom.pth".
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=str(RVC_ROOT / ".env"))

    from infer.modules.vc.modules import VC
    from configs.config import Config

    config = Config()
    vc = VC(config)
    vc.get_vc(MODEL_NAME)
    print(f"[RVC Server] Model '{MODEL_NAME}' loaded successfully.")
    USE_NATIVE = True
except ImportError:
    print("[RVC Server] Could not import RVC modules natively, using subprocess fallback.")
    USE_NATIVE = False
    import subprocess


# ─── Inference ───────────────────────────────────────────────────────────────

_lock = threading.Lock()  # RVC inference is not thread-safe

# Set by a {"command": "shutdown"} request. taskkill without /F cannot reach a
# console process parked in accept(), so an in-protocol shutdown is what allows
# this server to stop without a hard kill mid-inference.
_shutdown = threading.Event()
_server_sock = None


def convert(input_wav: str, output_wav: str) -> str:
    """Run voice conversion. Thread-safe via lock."""
    with _lock:
        if USE_NATIVE:
            # Direct Python call — no subprocess overhead.
            # NOTE: vc_single requires f0_file and file_index2 (no defaults),
            # and returns (info, (tgt_sr, audio)).
            info, (tgt_sr, audio) = vc.vc_single(
                sid=0,
                input_audio_path=input_wav,
                f0_up_key=0,
                f0_file=None,
                f0_method=F0_METHOD,
                file_index=INDEX_PATH,
                file_index2="",
                index_rate=INDEX_RATE,
                filter_radius=3,
                resample_sr=0,
                rms_mix_rate=0.25,
                protect=0.33,
            )
            if audio is None:
                raise RuntimeError(f"RVC inference failed: {info}")

            # Use the model's real sample rate (Mom.pth is 32k, not 40k).
            import soundfile as sf
            sf.write(output_wav, audio, tgt_sr)
        else:
            # Subprocess fallback (still faster than cold-starting every time
            # because this server stays warm)
            RVC_PYTHON = Path(r"D:\rvc\rvc_env\Scripts\python.exe")
            cmd = [
                str(RVC_PYTHON),
                "tools/infer_cli.py",
                "--model_name", MODEL_NAME,
                "--input_path", str(Path(input_wav).resolve()),
                "--opt_path", str(Path(output_wav).resolve()),
                "--f0method", F0_METHOD,
                "--index_path", INDEX_PATH,
                "--index_rate", str(INDEX_RATE),
            ]
            env = os.environ.copy()
            env["PATH"] = r"D:\rvc\rvc_env\Scripts;" + env["PATH"]
            subprocess.run(cmd, cwd=str(RVC_ROOT), check=True, env=env)

    return output_wav


# ─── TCP Server ──────────────────────────────────────────────────────────────

HOST = "127.0.0.1"
PORT = 7865


def handle_client(conn, addr):
    """Handle a single conversion request."""
    try:
        data = b""
        while True:
            chunk = conn.recv(4096)
            if not chunk:
                break
            data += chunk
            # Check if we have a complete JSON message
            try:
                request = json.loads(data.decode("utf-8"))
                break
            except json.JSONDecodeError:
                continue

        if not data:
            return

        request = json.loads(data.decode("utf-8"))

        # ── Control commands ──
        if request.get("command") == "shutdown":
            print("[RVC Server] Shutdown requested.")
            try:
                conn.sendall(json.dumps(
                    {"status": "ok", "message": "shutting down"}
                ).encode("utf-8"))
            except Exception:
                pass
            # Wait for any in-flight conversion to finish before tearing down,
            # so a half-written output WAV is never left behind.
            with _lock:
                pass
            _shutdown.set()
            if _server_sock is not None:
                try:
                    _server_sock.close()
                except Exception:
                    pass
            return

        if request.get("command") == "ping":
            conn.sendall(json.dumps({"status": "ok"}).encode("utf-8"))
            return

        input_path = request["input"]
        output_path = request["output"]

        print(f"[RVC Server] Converting: {input_path} → {output_path}")
        convert(input_path, output_path)

        response = json.dumps({"status": "ok", "output": output_path})
        conn.sendall(response.encode("utf-8"))
        print(f"[RVC Server] Done.")

    except Exception as e:
        error_resp = json.dumps({"status": "error", "message": str(e)})
        try:
            conn.sendall(error_resp.encode("utf-8"))
        except Exception:
            pass
        print(f"[RVC Server] Error: {e}")
    finally:
        conn.close()


def main():
    global _server_sock

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(2)
    _server_sock = server
    print(f"[RVC Server] Listening on {HOST}:{PORT}")
    print(f"[RVC Server] Model: {MODEL_NAME} | f0: {F0_METHOD}")
    print(f"[RVC Server] Ready for requests.\n")

    try:
        while not _shutdown.is_set():
            try:
                conn, addr = server.accept()
            except OSError:
                # Listener was closed by the shutdown handler.
                break
            # Handle in thread for non-blocking accept (but inference is serialized)
            t = threading.Thread(target=handle_client, args=(conn, addr), daemon=True)
            t.start()
    except KeyboardInterrupt:
        print("\n[RVC Server] Interrupted.")
    finally:
        try:
            server.close()
        except Exception:
            pass
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        print("[RVC Server] Stopped.")


if __name__ == "__main__":
    main()
