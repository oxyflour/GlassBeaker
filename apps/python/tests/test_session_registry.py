from __future__ import annotations

import asyncio
import unittest
from unittest import mock

from utils.session_registry import AsyncSessionRegistry


class _FakeSession:
    def __init__(self, active: bool = True) -> None:
        self._active = active

    def is_active(self) -> bool:
        return self._active


class AsyncSessionRegistryTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.registry = AsyncSessionRegistry[_FakeSession]()

    async def test_await_ready_evicts_failed_future_and_allows_retry(self):
        create = mock.AsyncMock(side_effect=[RuntimeError("boom"), _FakeSession()])

        first = self.registry.get_or_create("sess-1", create)
        with self.assertRaises(RuntimeError):
            await self.registry.await_ready("sess-1", first)
        self.assertNotIn("sess-1", self.registry.sessions)

        second = self.registry.get_or_create("sess-1", create)

        self.assertIs(self.registry.sessions["sess-1"], second)
        self.assertIsInstance(await self.registry.await_ready("sess-1", second), _FakeSession)

    def test_resolve_evicts_inactive_session(self):
        future: asyncio.Future[_FakeSession] = asyncio.Future()
        future.set_result(_FakeSession(active=False))
        self.registry.sessions["sess-1"] = future

        resolved, reason = self.registry.resolve("sess-1")

        self.assertIsNone(resolved)
        self.assertEqual(reason, "expired")
        self.assertNotIn("sess-1", self.registry.sessions)

    async def test_get_or_create_reuses_pending_future(self):
        gate = asyncio.Event()
        create = mock.AsyncMock(side_effect=lambda: gate.wait())

        first = self.registry.get_or_create("sess-1", create)
        second = self.registry.get_or_create("sess-1", create)

        self.assertIs(first, second)
        create.assert_called_once_with()
        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first
