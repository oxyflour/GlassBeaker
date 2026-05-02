# 21. RL Renderer Refactor Evaluation

更新时间：2026-05-02 14:39:17 +08:00

## 本次评估目标

评估以下重构方案是否成立：

- `mujoco` 不再作为 `apps/python` 的直接依赖
- Isaac renderer 尽量改为通过 `genie_sim` 自带运行方式启动
- 不修改 `deps/genie_sim` 上游源码

## 进展记录

### 14:37 第一轮核对

已核对文档与当前实现，确认 `doc/genie-sim/rl-renderer.md` 对现状描述基本准确，当前主链路仍然是：

1. `apps/python/api/zapdos/{session}/{action}.py`
2. `ensure_render_bundle(...)`
3. 本地 `mujoco` 会话
4. `apps/python/utils/sim_env.py` 启动 Isaac 子进程
5. `apps/isaac/rl_renderer_entry.py` 导入 `deps/genie_sim/source/geniesim/...`

### 14:37 已确认事实

1. `apps/python` 目前直接依赖 `mujoco`，不是只有一处：
   - `apps/python/api/zapdos/{session}/{action}.py`
   - `apps/python/utils/rl_bundle.py`
   - `apps/python/utils/sim_env.py`
   - `apps/python/utils/mujoco_tools.py`
   - `apps/python/tests/test_rl_bundle.py`
2. 当前 Isaac 启动并不是“直接使用 `genie_sim` 自带环境”：
   - `apps/python/utils/sim_env.py` 固定使用 `apps/isaac/.venv/Scripts/python.exe`
   - `apps/isaac/rl_renderer_entry.py` 只是把 `deps/genie_sim/source` 插入 `sys.path`
3. 上游 `genie_sim` 本身支持 MuJoCo / Isaac 分离启动：
   - `deps/genie_sim/source/geniesim/rl/envs/process_manager.py`
   - `isaac_python` 与 `mujoco_python` 是两个独立参数
4. 上游默认运行模型偏 Linux / Docker：
   - Isaac 入口默认 `/isaac-sim/python.sh`
   - `entrypoint_geniesim_rlinf.sh`、`dockerfile_geniesim_rlinf` 都围绕容器内环境准备
   - 这和当前仓库的 Windows + `uv` + `apps/isaac/.venv` 方案并不等价

## 当前判断

### 结论等级

当前方案“方向可行，但不是低成本重构”。

### 为什么不是低成本重构

如果目标只是“让 renderer 从更像 upstream 的入口启动”，改动较小。

如果目标是“`apps/python` 完全不安装 `mujoco`”，那就不是替换启动脚本，而是要重切进程边界，因为当前 `ZapdosSession` 里同时承担了：

- MuJoCo 模型加载
- joint command 应用
- joint state 读取
- TF 生成
- MuJoCo 几何导出给前端
- bundle 编译后的 MJCF 校验

这些能力现在都在 `apps/python` 进程内直接访问 `mujoco` API。

## 已识别的重构影响面

### A. 运行时链路

若移除 `apps/python` 对 `mujoco` 的直接依赖，至少要把以下能力迁到外部进程或服务：

- `ZapdosSession` 的 step loop
- `/call/get_visual`
- `/call/get_pose`
- joint command 到 MuJoCo 的写入
- joint state / tf 的读取

### B. bundle 编译链路

`ensure_render_bundle(...)` 目前会直接：

- `mujoco.MjModel.from_xml_path(...)`
- 遍历 body
- 前向计算 body pose

所以即使运行时搬空，`apps/python` 仍会因为 bundle 构建和测试继续依赖 `mujoco`，除非再拆一层“bundle 校验子进程”。

### C. 环境边界

上游 `genie_sim` 的“自带环境”本质上是：

- Isaac renderer 在 Isaac Python 下运行
- MuJoCo 节点在另一个 Python 下运行
- 主控进程负责编排

它不是一个可以直接替代 `apps/python` 的单解释器环境。

## 初步建议

### 推荐方向

推荐分两段做，而不是一步把 `mujoco` 从 `apps/python` 清空：

1. 先把 Isaac 启动进一步向 upstream 对齐
2. 再决定是否把 MuJoCo 会话下沉为独立子进程

### 第一阶段可接受目标

第一阶段只收敛到：

- `apps/isaac` 继续作为 Isaac 专属边界
- 启动参数、环境变量、入口脚本尽量贴近 `genie_sim`
- `apps/python` 仍保留 `mujoco`，避免一次性打散 `ZapdosSession`

这个阶段成本可控，也符合当前 `rl-renderer.md` 里已经形成的目录边界。

## 本轮聚焦问题

本轮重点判断的是：

- 是否值得直接复用 `geniesim.rl.envs.ProcessManager`
- 如果不直接复用，最小兼容改造面是什么
- 最终建议采用的迁移路径

## 14:39 第二轮结论

### 是否值得直接复用 `ProcessManager`

不建议直接复用为 `zapdos` 主链路。

原因：

1. `ProcessManager` 面向的是 RL 训练场景，不是 HTTP session 场景
2. 它默认搭配 `GenieSimVectorEnv` / `sim_server.py` / step SHM 协议工作
3. 当前 `zapdos` 前端依赖的接口是：
   - SSE `start`
   - `call/get_visual`
   - `render/main`
   - asset 文件直出
4. 这些接口和 upstream 的职责边界并不对齐
5. upstream 运行模型偏 Linux container，而当前项目有明确的 Windows 本地编排路径

结论：可以借鉴它的“MuJoCo 与 Isaac 分进程 + 独立 Python”思想，但不适合整块替换进来。

### 是否应该直接切到 `genie_sim` 自带启动

不建议直接切到 upstream 的完整启动方式。

更合适的表述应该是：

- 保持 `apps/isaac` 作为本仓库的 Isaac 边界
- 让 `apps/isaac` 的入口、参数和环境变量尽量贴近 upstream
- 继续通过本仓库已有的 `/api/isaac` 进程管理层启动

这样可以保留：

- Windows 可控性
- 当前日志与 SHM 管理方式
- 不修改上游源码的约束

同时减少：

- 容器依赖
- ROS/SHM 协议整体替换
- `zapdos` API 大改

## 推荐迁移路径

### Phase 1：只收敛 Isaac 边界

目标：

- `apps/python` 不再承担任何 Isaac-only 逻辑
- `apps/isaac/rl_renderer_entry.py` 继续做本地适配层
- 环境变量与参数组织尽量向 upstream 靠齐

这一步完成后，收益是目录边界更清楚，但 `apps/python` 仍保留 `mujoco`。

### Phase 2：如果确实要移除 `apps/python` 的 `mujoco`

前提是接受一次真正的架构切分：

- 把 MuJoCo session 改成独立子进程
- `apps/python` 改为 host / broker
- 通过 ROS、SHM 或轻量 RPC 从外部读取 pose / state / visual
- bundle 校验逻辑也要同步迁出或异步化

这一步才是“`mujoco` 不放在 `apps/python`”的真实成本。

## 最终建议

当前不建议把目标定义成：

- “马上移除 `apps/python` 的 `mujoco`”
- “直接改成 upstream `genie_sim` 原生启动”

建议把目标改成：

1. 先完成 Isaac 边界收敛
2. 保持 `apps/python` 内 MuJoCo 链路稳定
3. 等 `zapdos` 接口收敛后，再评估是否值得把 MuJoCo host 进程化

如果后续要继续推进，本任务下一步最合适的是先写一个更细的 Phase 1 改造清单，而不是直接动依赖。
