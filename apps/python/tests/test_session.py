from __future__ import annotations

import asyncio
import queue
import sys
import threading
import time
import unittest
from unittest import mock
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "apps" / "python"))

from utils.session import Session


class SessionTestCase(unittest.TestCase):
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


class SessionThreadingTest(unittest.IsolatedAsyncioTestCase):
    def _shutdown_session(self, session: Session) -> None:
        session.timeout = 0.001
        session.active = 0
        session.proc.join(timeout=1)

    async def test_run_sync_executes_callable_on_session_thread(self):
        class DummySession(Session):
            def step_once(self):
                time.sleep(0.001)

        session = DummySession(30)
        self.addCleanup(self._shutdown_session, session)
        for name in ("add_actor", "proc_actors", "has_actors"):
            self.assertNotIn(name, Session.__dict__)
            self.assertFalse(hasattr(session, name))

        caller_thread = threading.get_ident()
        session_thread, proc_thread = await asyncio.wait_for(
            session.run_sync(lambda current: (threading.get_ident(), current.proc.ident)),
            timeout=0.5,
        )

        self.assertEqual(session_thread, proc_thread)
        self.assertNotEqual(session_thread, caller_thread)

    async def test_long_running_call_keeps_session_active(self):
        started = threading.Event()
        release = threading.Event()

        class DummySession(Session):
            def call_once(self, method: str, args: tuple):
                del method, args
                started.set()
                release.wait(timeout=1)
                return "done"

            def step_once(self):
                time.sleep(0.001)

        session = DummySession(0.02)
        self.addCleanup(self._shutdown_session, session)

        pending = asyncio.create_task(session.call("block"))
        try:
            self.assertTrue(await asyncio.to_thread(started.wait, 0.5))
            await asyncio.sleep(0.05)

            self.assertTrue(session.is_active())
        finally:
            release.set()

        self.assertEqual(await asyncio.wait_for(pending, timeout=0.5), "done")

    async def test_reserve_world_sets_world_ownership_state(self):
        class DummySession(Session):
            def step_once(self):
                time.sleep(0.001)

        session = DummySession(30)
        self.addCleanup(self._shutdown_session, session)

        self.assertFalse(await session.run_sync(lambda current: current.world_owned()))

        async with session.reserve_world() as world_token:
            self.assertTrue(await session.run_sync(lambda current: current.world_owned(), world_token=world_token))

        self.assertFalse(await session.run_sync(lambda current: current.world_owned()))

    async def test_default_step_pauses_while_world_is_reserved(self):
        class DummySession(Session):
            def __init__(self, timeout=120) -> None:
                self.default_ticks = 0
                super().__init__(timeout)

            def step_once(self):
                self.default_ticks += 1
                time.sleep(0.001)

        session = DummySession(30)
        self.addCleanup(self._shutdown_session, session)

        while session.default_ticks == 0:
            await asyncio.sleep(0.001)

        async with session.reserve_world() as world_token:
            before = await session.run_sync(lambda current: current.default_ticks, world_token=world_token)
            await asyncio.sleep(0.02)
            after = await session.run_sync(lambda current: current.default_ticks, world_token=world_token)

        self.assertEqual(after, before)

        deadline = time.time() + 0.5
        while session.default_ticks == after and time.time() < deadline:
            await asyncio.sleep(0.001)

        self.assertGreater(session.default_ticks, after)

    async def test_call_waits_until_world_reservation_is_released(self):
        class DummySession(Session):
            def call_once(self, method: str, args: tuple):
                return {"method": method, "args": list(args)}

            def step_once(self):
                time.sleep(0.001)

        session = DummySession(30)
        self.addCleanup(self._shutdown_session, session)

        async with session.reserve_world():
            blocked = asyncio.create_task(session.call("ping", "value"))
            await asyncio.sleep(0.02)
            self.assertFalse(blocked.done())

        result = await asyncio.wait_for(blocked, timeout=0.5)
        self.assertEqual(result, {"method": "ping", "args": ["value"]})

    async def test_call_warns_when_method_takes_more_than_half_second(self):
        class DummySession(Session):
            def call_once(self, method: str, args: tuple):
                return {"method": method, "args": list(args)}

        session = DummySession.__new__(DummySession)
        session.calls = queue.Queue()
        session._deferred_calls = []
        session._world_owner = None
        session._running_calls = 0
        session.active = 0

        pending = asyncio.create_task(session.call("slow_method", "value"))
        await asyncio.sleep(0)

        with mock.patch("utils.session.time.time", side_effect=[10.0, 10.6]):
            with mock.patch("builtins.print") as print_mock:
                session.proc_once()

        self.assertEqual(
            await asyncio.wait_for(pending, timeout=0.5),
            {"method": "slow_method", "args": ["value"]},
        )
        print_mock.assert_called_once_with("WARN: session call slow_method took 0.600s")


class SessionLoopBindingTest(SessionTestCase):
    def test_schedule_on_owner_loop_uses_bound_loop_when_session_was_created_before_running_loop(self):
        class DummySession(Session):
            def step_once(self):
                time.sleep(0.001)

        session = self._create_session_before_asyncio_run(DummySession)
        seen: list[asyncio.AbstractEventLoop] = []

        async def marker():
            seen.append(asyncio.get_running_loop())
            return "ok"

        future = session.schedule_on_owner_loop(marker())
        asyncio.set_event_loop(session.loop)
        try:
            result = session.loop.run_until_complete(asyncio.wrap_future(future))
        finally:
            asyncio.set_event_loop(None)

        self.assertEqual(result, "ok")
        self.assertEqual(seen, [session.loop])

    def test_run_sync_resolves_when_session_was_created_before_running_loop(self):
        class DummySession(Session):
            def step_once(self):
                time.sleep(0.001)

        session = self._create_session_before_asyncio_run(DummySession)

        async def exercise():
            return await asyncio.wait_for(
                session.run_sync(lambda current: current.proc.ident),
                timeout=0.5,
            )

        result = asyncio.run(exercise())

        self.assertEqual(result, session.proc.ident)

    def test_run_sync_raises_when_session_was_created_before_running_loop(self):
        class DummySession(Session):
            def step_once(self):
                time.sleep(0.001)

        session = self._create_session_before_asyncio_run(DummySession)

        async def exercise():
            return await asyncio.wait_for(
                session.run_sync(lambda current: (_ for _ in ()).throw(RuntimeError("boom"))),
                timeout=0.5,
            )

        with self.assertRaisesRegex(RuntimeError, "boom"):
            asyncio.run(exercise())

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
