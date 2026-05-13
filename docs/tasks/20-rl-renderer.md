看 tasks\18-rl-renderer-run.md

继续结果：

- 根因不是 TF topic 本身，而是我们生成的 `sim_scene.usda` / `robot_wrapper.usda` / `scene_render.usda` / `render_scene.usda` 没有显式写 `upAxis`，新建 stage 默认回退成 `Y-up`。
- `apps/python/utils/usd_to_mjcf.py` 读取到这个错误的 `Y-up` 后，会额外插入 `usd_stage_root` 旋转补偿，导致 MuJoCo 产出的 body 世界位姿整体绕 X 轴偏了，Isaac wrapper 初始位姿也跟着错。
- 修复方式：在 `apps/python/utils/rl_bundle_stage.py` 里统一给所有生成 stage 写入来自 scene 的 stage metadata（当前是 `Z-up` + `metersPerUnit=1.0`），并 bump `BUNDLE_VERSION` 强制重编 bundle。
- 验证：
  - `apps/python/tests/test_rl_bundle.py` 已补 stage metadata 断言并通过。
  - 导出图像：
    - `apps/python/tmp/rl_debug/mujoco_pose_matched.png`
    - `apps/python/tmp/rl_debug/isaac_pose.png`
  - 当前 Isaac 图里机器人已恢复直立，不再出现之前那种整体姿态轴错位。
