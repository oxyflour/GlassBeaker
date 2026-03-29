加入 AI 生成代码，通过 sandpack 生成界面的功能

已完成
- 提供 frontend tool，允许 agent 生成 React 组件代码、CSS 和默认 props，并直接显示在 Sandpack 中
- 提供 frontend tool，允许 agent 只修改当前预览的 props，和用户共同围绕同一个组件继续交互
- 首页改为聊天 + Sandpack 预览工作台，用户可直接编辑 JSON props

后续可继续
- 增加更多 Sandpack 运行时依赖控制，例如允许 agent 声明额外 npm dependencies
- 为 props 编辑区补更友好的表单化编辑，而不只是一块 JSON 文本框
- 增加“从当前预览继续微调”的系统提示和更细的错误反馈
