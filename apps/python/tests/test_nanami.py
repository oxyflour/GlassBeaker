from __future__ import annotations

import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock
from pathlib import Path

from nanami.models import SourceConfig, manifest_from_dict
from nanami.stl_obj import stl_to_obj
from nanami.urdf import parse_robot_manifest
from nanami.worker import NanamiWorker


URDF_SAMPLE = """<robot name="demo">
  <link name="base_link"><visual><geometry><mesh filename="package://demo/meshes/base.STL" /></geometry></visual></link>
  <link name="left_arm_link1"><visual><material><color rgba="1 0 0 1" /></material><geometry><mesh filename="package://demo/meshes/a.STL" /></geometry></visual></link>
  <joint name="left_arm_joint1" type="revolute"><origin xyz="0 0 1" rpy="0 0 0" /><parent link="base_link" /><child link="left_arm_link1" /><axis xyz="0 1 0" /><limit lower="-1" upper="1" /></joint>
</robot>"""

ASCII_STL = """solid demo
facet normal 0 0 1
 outer loop
  vertex 0 0 0
  vertex 1 0 0
  vertex 0 1 0
 endloop
endfacet
endsolid demo
"""


class NanamiTest(unittest.TestCase):
    def test_manifest_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "robot.urdf"
            path.write_text(URDF_SAMPLE, encoding="utf-8")
            manifest = parse_robot_manifest(path, SourceConfig("repo", "ref", "R1"), "deadbeef")
            restored = manifest_from_dict(json.loads(json.dumps(manifest.to_dict())))
            self.assertEqual(restored.robot_id, "demo")
            self.assertEqual(restored.movable_joint_count, 1)
            self.assertEqual(restored.controls[0].name, "left_arm")

    def test_ascii_stl_to_obj(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "demo.stl"
            dest = Path(tmp) / "demo.obj"
            src.write_text(ASCII_STL, encoding="utf-8")
            stl_to_obj(src, dest)
            text = dest.read_text(encoding="utf-8")
            self.assertIn("v 1 0 0", text)
            self.assertIn("f 1//1 2//2 3//3", text)

    def test_worker_log_loop_decodes_utf8_stdout(self):
        events: list[dict] = []
        worker = NanamiWorker("rt1", Path("config.json"), events.append)
        worker.proc = SimpleNamespace(stdout=[b"log \xe2\x82\xac\n"]) # type: ignore
        expected = "log \u20ac"
        with mock.patch("nanami.worker.sys.stdout", new=SimpleNamespace(encoding="utf-8")):
            with mock.patch("builtins.print") as print_mock:
                worker._log_loop()
        print_mock.assert_called_once_with(f"[nanami:rt1] {expected}")
        self.assertEqual(events, [{"type": "worker_exit"}])
        worker.listener.close()

    def test_worker_log_loop_falls_back_to_preferred_encoding(self):
        events: list[dict] = []
        worker = NanamiWorker("rt2", Path("config.json"), events.append)
        expected = "\u4e2d\u6587"
        worker.proc = SimpleNamespace(stdout=[f"{expected}\n".encode("gbk")]) # type: ignore
        with mock.patch("nanami.worker.locale.getpreferredencoding", return_value="gbk"):
            with mock.patch("nanami.worker.sys.stdout", new=SimpleNamespace(encoding="utf-8")):
                with mock.patch("builtins.print") as print_mock:
                    worker._log_loop()
        print_mock.assert_called_once_with(f"[nanami:rt2] {expected}")
        self.assertEqual(events, [{"type": "worker_exit"}])
        worker.listener.close()

    def test_worker_log_loop_escapes_unencodable_console_text(self):
        events: list[dict] = []
        worker = NanamiWorker("rt3", Path("config.json"), events.append)
        worker.proc = SimpleNamespace(stdout=[b"log \xe2\x82\xac\n"]) # type: ignore
        with mock.patch("nanami.worker.sys.stdout", new=SimpleNamespace(encoding="gbk")):
            with mock.patch("builtins.print") as print_mock:
                worker._log_loop()
        print_mock.assert_called_once_with(r"[nanami:rt3] log \u20ac")
        self.assertEqual(events, [{"type": "worker_exit"}])
        worker.listener.close()

    def test_worker_stop_closes_streams_and_terminates_process(self):
        worker = NanamiWorker("rt4", Path("config.json"), lambda _event: None)
        worker.listener.close()
        listener = mock.Mock()
        file = mock.Mock()
        sock = mock.Mock()
        proc = mock.Mock()
        proc.poll.return_value = None
        worker.listener = listener
        worker.file = file
        worker.sock = sock
        worker.proc = proc

        worker.stop()

        file.close.assert_called_once_with()
        sock.shutdown.assert_called_once_with(mock.ANY)
        sock.close.assert_called_once_with()
        listener.close.assert_called_once_with()
        proc.terminate.assert_called_once_with()
        proc.wait.assert_called_once_with(timeout=5.0)

    def test_worker_send_buffers_until_socket_is_ready(self):
        worker = NanamiWorker("rt5", Path("config.json"), lambda _event: None)
        worker.send({"type": "hello"})

        self.assertEqual(len(worker.pending_payloads), 1)
        worker.listener.close()


if __name__ == "__main__":
    unittest.main()
