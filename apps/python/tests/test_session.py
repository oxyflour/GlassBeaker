from __future__ import annotations

import asyncio
import sys
import time
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "apps" / "python"))

from utils.session import Session


class SessionTest(unittest.TestCase):
    @unittest.skip("background Future queueing is unused in production")
    def test_background_future_queueing_is_unused(self):
        pass


class SessionLoopBindingTest(unittest.TestCase):
    def _shutdown_session(self, session: Session) -> None:
        session.timeout = 0.001
        session.active = 0
        session.proc.join(timeout=1)

    def _create_session_before_asyncio_run(self, factory):
        bootstrap_loop = asyncio.new_event_loop()
        self.addCleanup(bootstrap_loop.close)
        asyncio.set_event_loop(bootstrap_loop)
        try:
            session = factory()
        finally:
            asyncio.set_event_loop(None)
        self.addCleanup(self._shutdown_session, session)
        return session

    def test_call_resolves_when_session_was_created_before_running_loop(self):
        class DummySession(Session):
            def __init__(self) -> None:
                super().__init__(0)

            def call_once(self, method: str, args: tuple):
                return {"method": method, "args": list(args)}

            def step_once(self):
                time.sleep(0.001)

        session = self._create_session_before_asyncio_run(DummySession)

        async def exercise():
            return await asyncio.wait_for(session.call("publish", "/env_0/tf_render"), timeout=0.5)

        result = asyncio.run(exercise())

        self.assertEqual(result, {"method": "publish", "args": ["/env_0/tf_render"]})

    def test_call_raises_when_session_was_created_before_running_loop(self):
        class DummySession(Session):
            def __init__(self) -> None:
                super().__init__(0)

            def call_once(self, method: str, args: tuple):
                raise RuntimeError(f"{method} failed")

            def step_once(self):
                time.sleep(0.001)

        session = self._create_session_before_asyncio_run(DummySession)

        async def exercise():
            return await asyncio.wait_for(session.call("publish"), timeout=0.5)

        with self.assertRaisesRegex(RuntimeError, "publish failed"):
            asyncio.run(exercise())


if __name__ == "__main__":
    unittest.main()
