from __future__ import annotations

import os
import platform
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="GlassBeaker Python Service")


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/runtime")
async def runtime() -> dict[str, Any]:
    return {
        "service": "python",
        "python": platform.python_version(),
        "host": os.getenv("GLASSBEAKER_PYTHON_HOST", "127.0.0.1"),
        "port": int(os.getenv("GLASSBEAKER_PYTHON_PORT", "8000")),
    }


@app.api_route("/api/echo", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def echo(request: Request) -> JSONResponse:
    body: Any = None

    if request.method not in {"GET", "DELETE"}:
        raw_body = await request.body()
        if raw_body:
            try:
                body = await request.json()
            except Exception:
                body = raw_body.decode("utf-8", errors="replace")

    return JSONResponse(
        {
            "method": request.method,
            "path": str(request.url.path),
            "query": dict(request.query_params),
            "body": body,
        }
    )


def main() -> None:
    host = os.getenv("GLASSBEAKER_PYTHON_HOST", "127.0.0.1")
    port = int(os.getenv("GLASSBEAKER_PYTHON_PORT", "8000"))
    log_level = os.getenv("GLASSBEAKER_PYTHON_LOG_LEVEL", "info")

    uvicorn.run(app, host=host, port=port, log_level=log_level)


if __name__ == "__main__":
    main()
