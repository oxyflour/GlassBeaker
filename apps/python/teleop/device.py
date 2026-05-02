from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SpaceMouseSample:
    translation: tuple[float, float, float]
    rotation: tuple[float, float, float]
    buttons: tuple[bool, bool]


class SpaceMouseDevice:
    def __init__(self, module: Any | None = None, deadzone: float = 0.05) -> None:
        self._module = module
        self._resource = None
        self._device = None
        self.deadzone = deadzone
        self.connected = False
        self.last_error: str | None = None

    def poll(self) -> SpaceMouseSample | None:
        if self._device is None and not self._connect():
            return None
        try:
            state = self._device.read()
        except Exception as exc:
            self.last_error = str(exc)
            self.close()
            return None
        if state is None:
            return None
        self.connected = True
        return SpaceMouseSample(
            translation=self._vector(state, (("x",), ("y",), ("z",))),
            rotation=self._vector(state, (("roll", "rx"), ("pitch", "ry"), ("yaw", "rz"))),
            buttons=self._buttons(state),
        )

    def status(self) -> dict[str, object]:
        return {"connected": self.connected, "last_error": self.last_error}

    def close(self) -> None:
        try:
            if self._resource is not None:
                self._resource.__exit__(None, None, None)
            elif self._device is not None and hasattr(self._device, "close"):
                self._device.close()
        except Exception:
            pass
        self._resource = None
        self._device = None
        self.connected = False

    def _connect(self) -> bool:
        try:
            module = self._module
            if module is None:
                import pyspacemouse as module  # type: ignore
            resource = module.open()
            if hasattr(resource, "__enter__") and hasattr(resource, "__exit__"):
                self._resource = resource
                self._device = resource.__enter__()
            else:
                self._device = resource
                self._resource = None
            self.connected = True
            self.last_error = None
            return True
        except Exception as exc:
            self.last_error = str(exc)
            self.connected = False
            return False

    def _vector(self, state: object, names: tuple[tuple[str, ...], ...]) -> tuple[float, float, float]:
        values = [self._axis(state, options) for options in names]
        return tuple(self._apply_deadzone(value) for value in values)  # type: ignore[return-value]

    def _axis(self, state: object, options: tuple[str, ...]) -> float:
        for name in options:
            value = getattr(state, name, None)
            if value is not None:
                return float(value)
        return 0.0

    def _buttons(self, state: object) -> tuple[bool, bool]:
        buttons = getattr(state, "buttons", None)
        if isinstance(buttons, (list, tuple)):
            left = bool(buttons[0]) if len(buttons) > 0 else False
            right = bool(buttons[1]) if len(buttons) > 1 else False
            return left, right
        return bool(getattr(state, "left", False)), bool(getattr(state, "right", False))

    def _apply_deadzone(self, value: float) -> float:
        return 0.0 if abs(value) < self.deadzone else value
