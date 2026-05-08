from __future__ import annotations

import asyncio
import unittest
from concurrent.futures import Future

from utils.session import Session


class _BackgroundSession(Session):
    def __init__(self) -> None:
        self.results: dict[str, Future[str]] = {}
        super().__init__(timeout=5)

    def call_once(self, method: str, args: tuple):
        if method == "background":
            future: Future[str] = Future()
            self.results["background"] = future
            return future
        if method == "ping":
            return "pong"
        return super().call_once(method, args)


class SessionTest(unittest.IsolatedAsyncioTestCase):
    async def test_call_waits_for_background_future_and_keeps_queue_moving(self):
        session = _BackgroundSession()

        try:
            background = asyncio.create_task(session.call("background"))
            for _ in range(50):
                if "background" in session.results:
                    break
                await asyncio.sleep(0.01)
            self.assertIn("background", session.results)
            ping = await asyncio.wait_for(session.call("ping"), timeout=2)
            session.results["background"].set_result("done")
            done = await asyncio.wait_for(background, timeout=2)
        finally:
            session.timeout = 0
            session.proc.join(timeout=1)

        self.assertEqual(ping, "pong")
        self.assertEqual(done, "done")


if __name__ == "__main__":
    unittest.main()
