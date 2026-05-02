from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "deps" / "genie_sim" / "source"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
os.environ.setdefault("SIM_REPO_ROOT", str(REPO_ROOT / "deps" / "genie_sim"))

import geniesim.rl.renderer.rl_renderer as upstream  # type: ignore


def _noop_spin(self) -> None:
    return None


upstream.rclpy.executors.SingleThreadedExecutor.spin = _noop_spin


class LocalRLRenderer(upstream.RLRenderer):
    def _create_default_viz_camera(self, env_path: str, cam_pos, cam_target) -> str:
        existing_path = env_path + "/default_viz_camera"
        if upstream.is_prim_path_valid(existing_path):
            return existing_path
        return super()._create_default_viz_camera(env_path, cam_pos, cam_target)

    def run(self) -> None:
        while upstream.simulation_app.is_running():
            self._ros_executor.spin_once(timeout_sec=0.0)
            self.world.step(render=True)
        self._ros_executor.shutdown(timeout_sec=2.0)
        for sub in self.env_subscribers:
            sub.destroy_node()
        upstream.rclpy.shutdown()
        if self.shm:
            self.shm.close()
            self.shm.unlink()
        self.world.stop()
        upstream.simulation_app.close()


if __name__ == "__main__":
    renderer = LocalRLRenderer(upstream._args)
    renderer.setup()
    renderer.run()
