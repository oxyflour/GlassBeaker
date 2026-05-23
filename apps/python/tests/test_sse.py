from __future__ import annotations

import json
import unittest

from fastapi.responses import StreamingResponse

from utils.sse import sse_json_event, sse_response


class SseHelpersTest(unittest.TestCase):
    def test_sse_json_event_encodes_named_json_event(self):
        self.assertEqual(
            sse_json_event("progress", {"stage": "prepare", "ok": True}),
            'event: progress\ndata: {"stage": "prepare", "ok": true}\n\n',
        )

    def test_sse_response_sets_event_stream_headers(self):
        async def events():
            yield "progress", {"stage": "prepare"}

        response = sse_response(events())

        self.assertIsInstance(response, StreamingResponse)
        self.assertEqual(response.media_type, "text/event-stream")
        self.assertEqual(response.charset, "utf-8")
        self.assertEqual(response.headers["cache-control"], "no-cache, no-transform")
        self.assertEqual(response.headers["x-accel-buffering"], "no")


if __name__ == "__main__":
    unittest.main()
