from __future__ import annotations

from contextlib import suppress
import json
import locale
import os
import socket
import subprocess
import sys
import threading
from pathlib import Path

from .config import DEFAULT_ENGINE_ROOT, HDR_PATH, SESSION_ENV, UPROJECT_PATH, WORKER_BOOTSTRAP


def _decode_log_line(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode(locale.getpreferredencoding(False), errors="replace")


def _console_safe_text(text: str) -> str:
    encoding = getattr(sys.stdout, "encoding", None) or locale.getpreferredencoding(False)
    return text.encode(encoding, errors="backslashreplace").decode(encoding)


class NanamiWorker:
    def __init__(self, session, config_path: Path, on_event):
        self.session = session
        self.config_path = config_path
        self.on_event = on_event
        self.proc: subprocess.Popen[bytes] | None = None
        self.sock: socket.socket | None = None
        self.file = None
        self.writer_lock = threading.Lock()
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(1)
        self.port = self.listener.getsockname()[1]

    def start(self) -> None:
        if not DEFAULT_ENGINE_ROOT.exists():
            raise FileNotFoundError(f"Unreal engine root not found: {DEFAULT_ENGINE_ROOT}")
        env = {**os.environ, SESSION_ENV: str(self.config_path)}
        exe = DEFAULT_ENGINE_ROOT / "Engine" / "Binaries" / "Win64" / "UnrealEditor-Cmd.exe"
        cmd = [
            str(exe),
            str(UPROJECT_PATH),
            f"-ExecutePythonScript={WORKER_BOOTSTRAP}",
            "-unattended",
            "-stdout",
            "-FullStdOutLogOutput",
            "-NoSplash",
            "-NoSound",
        ]
        print(f"[nanami:{self.session.session_id}] Starting worker: {exe}")
        print(f"[nanami:{self.session.session_id}] Command: {' '.join(cmd)}")
        self.proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(f"[nanami:{self.session.session_id}] Worker process started, pid={self.proc.pid}")
        threading.Thread(target=self._accept_loop, daemon=True).start()
        threading.Thread(target=self._log_loop, daemon=True).start()

    def _log_loop(self) -> None:
        if not self.proc or not self.proc.stdout:
            return
        for raw in self.proc.stdout:
            line = _decode_log_line(raw).rstrip()
            print(_console_safe_text(f"[nanami:{self.session.session_id}] {line}"))
        self.on_event({"type": "worker_exit"})

    def _accept_loop(self) -> None:
        print(f"[nanami:{self.session.session_id}] Waiting for worker connection on port {self.port}...")
        conn, addr = self.listener.accept()
        print(f"[nanami:{self.session.session_id}] Worker connected from {addr}")
        self.sock = conn
        self.file = conn.makefile("r", encoding="utf-8")
        self.session.attach_command_sink(self.send)
        for raw in self.file:
            self.on_event(json.loads(raw))

    def send(self, payload: dict) -> None:
        if not self.sock:
            return
        data = (json.dumps(payload) + "\n").encode("utf-8")
        with self.writer_lock:
            self.sock.sendall(data)

    def write_session_config(self, payload: dict) -> None:
        payload.update(
            {
                "control_host": "127.0.0.1",
                "control_port": self.port,
                "hdr_path": str(HDR_PATH),
            }
        )
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def stop(self, timeout: float = 5.0) -> None:
        if self.file is not None:
            with suppress(OSError):
                self.file.close()
            self.file = None
        if self.sock is not None:
            with suppress(OSError):
                self.sock.shutdown(socket.SHUT_RDWR)
            with suppress(OSError):
                self.sock.close()
            self.sock = None
        with suppress(OSError):
            self.listener.close()
        if not self.proc or self.proc.poll() is not None:
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=timeout)
