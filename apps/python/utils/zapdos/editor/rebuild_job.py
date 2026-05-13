from __future__ import annotations

import queue
from concurrent.futures import Future as ConcurrentFuture
from dataclasses import dataclass


@dataclass
class SceneRebuildJob:
    future: ConcurrentFuture
    success_payload: dict[str, object]
    events: queue.Queue[tuple[str, dict[str, object]]]
