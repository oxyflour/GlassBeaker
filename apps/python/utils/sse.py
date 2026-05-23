from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi.responses import StreamingResponse

SSE_MEDIA_TYPE = "text/event-stream"
SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def sse_json_event(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _encode_events(events: AsyncIterator[tuple[str, Any]]) -> AsyncIterator[str]:
    async for event, data in events:
        yield sse_json_event(event, data)


def sse_response(events: AsyncIterator[tuple[str, Any]]) -> StreamingResponse:
    return StreamingResponse(
        _encode_events(events),
        media_type=SSE_MEDIA_TYPE,
        headers=SSE_HEADERS,
    )
