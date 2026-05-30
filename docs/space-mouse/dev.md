# SpaceMouse 线程版接入（Zapdos，单活跃臂切换）

## Summary
- 不开新进程，也不把控制线程挂到 `ZapdosSession`。改为 `apps/python` 服务内的全局单例 `SpaceMouseManager`，按 API 懒启动，独立于任意 Zapdos session 生命周期。
- 左右臂都支持，但任一时刻只控制一只手；默认活跃臂为 `right`，运行时只通过 API 切换。
- 线程内部读取 SpaceMouse，订阅 `/env_0/joint_states` 回填当前姿态，在本地 MuJoCo 模型上做 FK/IK，再发布 `/env_0/joint_command`。`apps/web` 和现有 Zapdos 路由保持不变。

## File Changes
- 修改 [apps/python/pyproject.toml](/c:/Projects/GlassBeaker/apps/python/pyproject.toml)
  增加 SpaceMouse/HID 与 WebSocket 依赖。
- 新增 `apps/python/api/teleop/spacemouse.py`
  用 `APIRouter` 暴露控制接口，并在 router shutdown 时停止全局 manager。
- 新增 `apps/python/teleop/manager.py`
  全局单例、后台线程、状态机、活跃臂切换、启动/停止。
- 新增 `apps/python/teleop/device.py`
  SpaceMouse 6 轴与按钮读取、死区、缩放、断连恢复。
- 新增 `apps/python/teleop/ros_client.py`
  连接本地 `/api/ros/ws`，订阅 `joint_states`，发布 `joint_command`。
- 新增 `apps/python/teleop/arm_config.py`
  固定左右臂命名映射：
  `left_arm_joint1..7` / `right_arm_joint1..7`
  `left_gripper_finger_joint1/2` / `right_gripper_finger_joint1/2`
  `Root_r1_pro_with_gripper_left_gripper_link` / `...right_gripper_link`
- 新增 `apps/python/teleop/ik_controller.py`
  MuJoCo 模型加载、关节状态同步、阻尼最小二乘 IK、限位与步长裁剪。
- 新增测试文件：
  `apps/python/tests/test_spacemouse_api.py`
  `apps/python/tests/test_spacemouse_manager.py`
  `apps/python/tests/test_spacemouse_ik.py`
  `apps/python/tests/test_spacemouse_arm_config.py`
- 保持不改：
  [apps/python/app.py](/c:/Projects/GlassBeaker/apps/python/app.py)
  [apps/python/api/zapdos/{session}/{action}.py](/c:/Projects/GlassBeaker/apps/python/api/zapdos/%7Bsession%7D/%7Baction%7D.py)
  [apps/web/app/demo/zapdos/page.tsx](/c:/Projects/GlassBeaker/apps/web/app/demo/zapdos/page.tsx)

## Interfaces And Behavior
- 新接口固定为：
  `POST /python/teleop/spacemouse/start`
  `POST /python/teleop/spacemouse/stop`
  `GET /python/teleop/spacemouse/status`
  `POST /python/teleop/spacemouse/set_active_arm`
- `start` 接收：
  `robot_usd`、`scene_usd`、`rate_hz`、`linear_scale`、`angular_scale`、`gripper_step`
- `set_active_arm` 只接受 `left` 或 `right`。
- 切换活跃臂时，新活跃臂目标立即吸附到该臂当前实际末端姿态，不继承另一只手的旧目标。
- 按钮语义固定：
  左键张开当前活跃臂夹爪
  右键闭合当前活跃臂夹爪
  双键同时按下将当前活跃臂目标重置到当前实际姿态
- 平移按 `base_link` 坐标系解释；姿态按当前活跃末端局部坐标系增量解释。
- 不复用 `utils.session.Session`，避免 120 秒 idle timeout 与 Zapdos session 生命周期耦合。
- 不直接操作 `utils.ros_bridge.bridge` 全局对象；线程内用独立 WebSocket client 走现有 `/api/ros/ws` 协议。

## Test Plan
- `arm_config` 测试确认左右臂、夹爪、末端 link 命名与 `deps/galaxea/object/r1pro/r1pro.xml` 一致。
- API 测试覆盖 `start/stop/status/set_active_arm`，并验证单例行为。
- manager 测试覆盖活跃臂切换、按钮语义、设备断开与 ROS 断连后的安全空闲状态。
- IK 测试分别对左臂和右臂做小幅末端增量，验证误差下降且 joint 不越界；特别覆盖左右臂非对称 joint2 限位。
- 手工联调确认：
  Zapdos 运行时，SpaceMouse 可驱动右臂；
  切到 `left` 后只驱动左臂；
  非活跃臂保持不动；
  页面渲染与相机流无改动。

## Assumptions
- v1 只支持 `/env_0/*` 单会话 ROS 话题，不做多 session 隔离。
- v1 不做双臂同时控制，不做实机安全链路。
- 设备输入走 Python HID 方案，不引入 Windows 上缺失的 `spacenav` 依赖。
- 线程由 teleop API 懒启动；服务启动时不自动运行。
