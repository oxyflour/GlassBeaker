from __future__ import annotations

import os
import platform
from typing import Any

import uvicorn
from fastapi import FastAPI
from ag_ui_langgraph import add_langgraph_fastapi_endpoint
from pathlib import Path

from utils.mount import mount_routes
from utils.module import load_module

# FIXME: https://github.com/CopilotKit/CopilotKit/issues/3279
import json
json_encode = json.JSONEncoder.default
def default_serialization(self, o):
    if o.__class__.__name__ == 'Context':
        return '{}'
    return json_encode(self, o)
json.JSONEncoder.default = default_serialization

# watch dog
import sys, threading
def wait_for_stdin():
    sys.stdin.readline()
    os._exit(0)
threading.Thread(target=wait_for_stdin, daemon=True).start()

app = FastAPI(title="GlassBeaker Python Service")
mount_routes(app, 'api')

agents = []
agent_root = Path(os.path.normpath(f'{__file__}/../agents'))
for item in os.listdir(agent_root):
    agent = None
    if item.endswith('.py'):
        module = load_module(agent_root / item)
        agent = getattr(module, 'agent', None)
    if agent:
        path = f"/agent/{item}"
        agents.append({ 'path': path, 'name': item.replace('.py', '') })
        add_langgraph_fastapi_endpoint(app=app, agent=agent, path=path)

@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/runtime")
async def runtime() -> dict[str, Any]:
    return {
        "service": "python",
        "python": platform.python_version(),
        "agents": agents
    }


def main() -> None:
    port = int(os.getenv("LISTEN_PORT", "13001"))
    log_level = os.getenv("LOG_LEVEL", "info")
    uvicorn.run(app, port=port, log_level=log_level)

if __name__ == "__main__":
    main()
