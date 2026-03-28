from __future__ import annotations

import os
import platform
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

import sys, threading
def wait_for_stdin():
    sys.stdin.readline()
    os._exit(0)
threading.Thread(target=wait_for_stdin, daemon=True).start()

app = FastAPI(title="GlassBeaker Python Service")


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
    host = os.getenv("GLASSBEAKER_PYTHON_HOST", "127.0.0.1")
    port = int(os.getenv("GLASSBEAKER_PYTHON_PORT", "4000"))
    log_level = os.getenv("GLASSBEAKER_PYTHON_LOG_LEVEL", "info")
    uvicorn.run(app, host=host, port=port, log_level=log_level)

if __name__ == "__main__":
    main()
