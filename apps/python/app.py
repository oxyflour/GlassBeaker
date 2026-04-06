from __future__ import annotations

import os
import platform
from typing import Any

import uvicorn
from fastapi import FastAPI

from utils.mount import mount_routes

# watch dog
import sys, threading
def wait_for_stdin():
    sys.stdin.readline()
    os._exit(0)
threading.Thread(target=wait_for_stdin, daemon=True).start()

app = FastAPI(title="GlassBeaker Python Service")
mount_routes(app, 'api')


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/runtime")
async def runtime() -> dict[str, Any]:
    return {
        "service": "python",
        "python": platform.python_version(),
    }

def main() -> None:
    port = int(os.getenv("LISTEN_PORT", "13001"))
    log_level = os.getenv("LOG_LEVEL", "info")
    uvicorn.run(app, port=port, log_level=log_level)

if __name__ == "__main__":
    main()
