为了减小维护负担，不要直接用 pi-web-ui 里的 ChatPanel，用它的底层组件重新设计 ui
1. pi-coding-agent 吐出的事件不要再封装，直接通过 sse 喂到前端
2. 在前端通过 pi-web-ui 的 message 等基础事件进行交互
