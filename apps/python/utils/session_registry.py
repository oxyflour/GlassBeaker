from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Protocol, TypeVar


class ActiveSession(Protocol):
    def is_active(self) -> bool: ...


TSession = TypeVar("TSession", bound=ActiveSession)


class AsyncSessionRegistry[TSession]:
    def __init__(self) -> None:
        self.sessions: dict[str, asyncio.Future[TSession]] = {}

    def discard(self, session_id: str, future: asyncio.Future[TSession]) -> None:
        if self.sessions.get(session_id) is future:
            self.sessions.pop(session_id, None)

    def resolve(self, session_id: str) -> tuple[asyncio.Future[TSession] | None, str | None]:
        future = self.sessions.get(session_id)
        if future is None:
            return None, None
        if future.cancelled():
            self.discard(session_id, future)
            return None, "missing"
        if not future.done():
            return future, None
        if future.exception() is not None:
            self.discard(session_id, future)
            return None, "missing"
        if not future.result().is_active():
            self.discard(session_id, future)
            return None, "expired"
        return future, None

    def get_or_create(
        self,
        session_id: str,
        create_session: Callable[[], Awaitable[TSession]],
    ) -> asyncio.Future[TSession]:
        future, _ = self.resolve(session_id)
        if future is not None:
            return future
        future = asyncio.create_task(create_session())
        self.sessions[session_id] = future
        return future

    async def await_ready(
        self,
        session_id: str,
        future: asyncio.Future[TSession],
    ) -> TSession:
        try:
            return await future
        except Exception:
            self.discard(session_id, future)
            raise
