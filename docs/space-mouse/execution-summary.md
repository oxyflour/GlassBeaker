# SpaceMouse 执行摘要

## 目标
- 在不改 `apps/web` 和现有 Zapdos 路由的前提下，把 SpaceMouse 接入 `apps/python`。
- 通过独立 teleop API 懒启动后台线程，驱动 `/env_0/joint_command`。
- 支持左右臂单活跃切换，默认 `right`。

## 当前结构
- `apps/python/api/teleop/spacemouse.py`
  对外提供 `start`、`stop`、`status`、`set_active_arm`。
- `apps/python/teleop/manager.py`
  负责线程循环、目标姿态、夹爪按钮语义和 IK 调用。
- `apps/python/teleop/ros_client.py`
  通过 `/api/ros/ws` 订阅 `joint_states`，发布 `joint_command`。
- `apps/python/teleop/device.py`
  读取 SpaceMouse 六轴和按键。

## 已确认阻塞
1. `POST /api/teleop/spacemouse/start` 空请求会把 `robot_usd` 和 `scene_usd` 覆盖成 `null`。
2. `SpaceMouseManager` 只有拿到 `/env_0/joint_states` 才会继续执行，所以 ROS 桥不通时不会进入真正控制环。
3. `teleop/ros_client.py` 和现有 `/api/ros/ws` 协议疑似不匹配，需要用测试确认并修正。
4. 当前仓库里没有前端自动触发 `spacemouse/start`，只能先靠后端 API 手动启动验证。

## 执行顺序
1. 先修 `start` 默认参数处理，保证无参启动不会坏掉。
2. 再用测试钉住 ROS websocket 桥接行为，修复 `ros_client` 与现有桥接协议的兼容性。
3. 完成后跑聚焦测试，并重新检查：
   - `running`
   - `ros_connected`
   - `device_connected`
   - `last_joint_state_at`
4. 最后再做一次手工联调，确认 SpaceMouse 真实输入能驱动 Zapdos。

## 本轮不做
- 不改 `apps/web` 页面交互。
- 不做双臂同时控制。
- 不做多 session ROS 隔离。
