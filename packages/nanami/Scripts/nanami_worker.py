from __future__ import annotations

import json
import os
import queue
import socket
import threading
import traceback
from pathlib import Path

import unreal  # type: ignore

from nanami_runtime import RobotRuntime


class Worker:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.sock = socket.create_connection((cfg["control_host"], cfg["control_port"]))
        self.file = self.sock.makefile("r", encoding="utf-8")
        self.commands: queue.Queue[dict] = queue.Queue()
        self.tick_handle = None
        self.runtime = RobotRuntime(cfg, self.send)

    def send(self, event: dict) -> None:
        payload = {**event, "runtime_id": self.cfg["runtime_id"]}
        self.sock.sendall((json.dumps(payload) + "\n").encode("utf-8"))

    def start(self) -> None:
        unreal.log("[Nanami] Worker starting...")
        unreal.EditorPythonScripting.set_keep_python_script_alive(True)
        self.runtime.setup_scene()
        threading.Thread(target=self._read_loop, daemon=True).start()
        self.tick_handle = unreal.register_slate_post_tick_callback(self._tick)
        self.send({"type": "hello"})

    def _read_loop(self) -> None:
        for raw in self.file:
            self.commands.put(json.loads(raw))

    def _tick(self, _delta_time: float) -> None:
        while not self.commands.empty():
            try:
                self._handle(self.commands.get())
            except Exception:
                self.send({"type": "worker_error", "message": "".join(traceback.format_exc())})

    def _handle(self, command: dict) -> None:
        kind = command.get("type")
        if kind == "attach_session":
            self.runtime.attach_session(command["session_id"], command["preview_dir"])
        elif kind == "detach_session":
            self.runtime.detach_session(command["session_id"])
        elif kind == "load_robot":
            self.runtime.load_robot(command["manifest"])
        elif kind == "set_state":
            self.runtime.apply_state(command["session_id"], command.get("joints", {}), command.get("camera"))
        elif kind == "export_final":
            self.runtime.export_final(command["session_id"], command["job_id"], command["output_dir"])

    def stop(self) -> None:
        if self.tick_handle is not None:
            unreal.unregister_slate_post_tick_callback(self.tick_handle)
        unreal.EditorPythonScripting.set_keep_python_script_alive(False)


def main():
    cfg_path = Path(os.environ["UE_NANAMI_SESSION_CONFIG"])
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    worker = Worker(cfg)
    unreal.register_python_shutdown_callback(worker.stop)
    worker.start()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        unreal.log_error("[Nanami] Fatal worker exception:\n" + "".join(traceback.format_exc()))
        try:
            unreal.EditorPythonScripting.set_keep_python_script_alive(False)
        except Exception:
            pass
        raise
