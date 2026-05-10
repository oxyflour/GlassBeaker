from __future__ import annotations

from multiprocessing import shared_memory

import numpy as np

SHM_HEADER_BYTES = 4


class SharedFrameBuffer:
    def __init__(self, shm_name: str, num_cameras: int, width: int, height: int) -> None:
        self.shm_name = shm_name
        self.num_cameras = num_cameras
        self.width = width
        self.height = height
        self.shm: shared_memory.SharedMemory | None = None
        self.frame_counter: np.ndarray | None = None
        self.frames: np.ndarray | None = None

    def bind(self) -> None:
        if self.shm is not None:
            return
        self.shm = shared_memory.SharedMemory(name=self.shm_name)
        self.frame_counter = np.ndarray((1,), dtype=np.uint32, buffer=self.shm.buf, offset=0)
        self.frames = np.ndarray(
            (1, self.num_cameras, self.height, self.width, 3),
            dtype=np.uint8,
            buffer=self.shm.buf,
            offset=SHM_HEADER_BYTES,
        )

    def read(self, camera_index: int) -> tuple[int, np.ndarray] | None:
        self.bind()
        if self.frame_counter is None or self.frames is None:
            return None
        return int(self.frame_counter[0]), self.frames[0, camera_index].copy()

    def close(self) -> None:
        if self.shm is not None:
            self.shm.close()
            self.shm = None
            self.frame_counter = None
            self.frames = None
