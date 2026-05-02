# RL Renderer 实现要点

## 背景

目标是让 `apps/isaac/genie_r1pro_demo.py` 驱动的 MuJoCo `tf_render` 数据，稳定同步到 RL renderer 的 USD 画面里，并满足新的约束：

- 不修改 `deps/genie_sim` 源码
- 由工作区外层代码和资源去适配 `genie_sim`

## 为什么一开始没同步

一开始的现象是：

- MuJoCo 侧在持续发布 `/env_0/tf_render`
- renderer 画面基本不动，像是“没同步上”

后来拆开链路看，实际不是单点问题，而是三个条件叠在一起：

### 1. ROS 回调没有稳定跟上渲染主循环

在这套 Windows + Isaac 环境里，上游 renderer 默认用后台 ROS executor。实际运行时，`world.step(render=True)` 会让后台回调线程不稳定，导致 TF 消息即使已经发布，renderer 侧也不一定能按预期及时消费。

结果是：

- MuJoCo 侧有 `tf_render`
- renderer 侧却可能没有稳定执行对应的 TF 回调和 prim 更新

### 2. body map 没有按上游的实际查找方式命中

上游 renderer 会根据 `scene_usd` 去推导 sidecar body map 路径，但对 `.usda` 场景文件存在实际兼容性问题，最终会去找：

- `r1pro_demo_scene_body_map.jsona`

而不是我们一开始只生成的：

- `r1pro_demo_scene_body_map.json`

这导致 renderer 没有读到 body map，后续只能回退到默认 body 名推断。

### 3. 最初的 wrapper 结构也不符合上游默认假设

在没有命中 body map 时，上游会按 MuJoCo body 名直接拼 prim 路径。它隐含假设 wrapper 是“每个 link 直接可写 world pose”的扁平结构。

但最初外层生成的 wrapper 更接近真实机器人层级，例如：

- `/MyRobot/r1_pro_with_gripper/right_arm_link1`

而上游默认会去找类似：

- `/World/envs/env_0/Root_r1_pro_with_gripper_right_arm_link1`

这两者对不上，所以即使 TF 到了，也找不到正确 prim 去应用位姿。

## 后来怎么解决

最终方案没有保留对 `genie_sim` 源码的修改，而是把适配全部搬到 `apps/isaac`。

### 1. 用本地入口脚本接管 renderer 启动

文件：

- `apps/isaac/rl_renderer_entry.py`

做法：

- 仍然导入上游 `geniesim.rl.renderer.rl_renderer`
- 只在运行时 patch executor 的驱动方式
- 让 ROS `spin_once` 跟 renderer 主循环一起推进

这样做的作用是：

- 不改 `deps/genie_sim` 文件内容
- 但把“回调跟不上渲染循环”的问题收敛在外层入口解决

### 2. 把 wrapper 改成上游能直接消费的扁平 per-link prim

文件：

- `apps/isaac/r1pro_rl_wrapper.py`

做法：

- 生成扁平 link prim，而不是保留原先那种机器人层级树
- prim 名字直接对齐 MuJoCo body 名约定
- 例如把 link 变成 `Root_r1_pro_with_gripper_<link>`

这样即使上游按默认 body 名推 prim 路径，也能命中实际 prim。

### 3. 同时生成正常版和兼容版 body map

文件：

- `apps/isaac/genie_r1pro_demo_support.py`
- `apps/isaac/r1pro_demo_scene_body_map.json`
- `apps/isaac/r1pro_demo_scene_body_map.jsona`

做法：

- 正常生成 `r1pro_demo_scene_body_map.json`
- 额外再生成一份 `r1pro_demo_scene_body_map.jsona`

这样即使不改上游路径推导逻辑，renderer 也能读到外层提供的 body map。

### 4. 由 demo support 统一接线

文件：

- `apps/isaac/genie_r1pro_demo_support.py`

负责：

- 调用 wrapper 生成逻辑
- 生成 body map 及兼容文件
- 把 renderer 启动入口切到本地 `rl_renderer_entry.py`

也就是说，真正的落地策略不是“修上游”，而是“把上游当成黑盒，补齐它的输入前提和运行时约束”。

## 实现后的效果

验证结果包括：

- `deps/genie_sim` 工作树保持干净
- renderer 共享内存前后帧有明显差异
- 实测帧差结果：
  - `early_counter = 5`
  - `late_counter = 50`
  - `mean_abs_diff = 4.0361`
  - `max_diff = 96`
  - `changed_pixels_gt10 = 3950`

这说明最终不是“只有 MuJoCo 在动”，而是 renderer 画面也已经跟着 TF 更新发生了明显变化。

## 结论

这次实现的关键，不是继续修改 `genie_sim`，而是明确上游 renderer 的几个隐含假设，然后在外层逐一满足：

- 运行时要保证 ROS 回调能跟上渲染循环
- scene sidecar 要兼容上游实际使用的 body map 路径
- robot wrapper 要兼容上游直接按 body 名写 prim world pose 的方式

满足这三点后，RL renderer 同步链路就能在不修改 `genie_sim` 源码的前提下跑通。
