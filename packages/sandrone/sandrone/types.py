from dataclasses import dataclass
from typing import Literal


Point = tuple[int, int]
Side = Literal["top", "bottom"]
KeepoutKind = Literal["polygon", "wire"]


@dataclass(frozen=True)
class Pin:
    pin_id: str
    group_id: str
    side: Side
    x: int
    y: int

    @property
    def point(self) -> Point:
        return (self.x, self.y)


@dataclass(frozen=True)
class PowerGroup:
    group_id: str
    marker: str
    layer: str
    pins: tuple[Pin, ...]


@dataclass(frozen=True)
class KeepoutShape:
    layer: str
    kind: KeepoutKind
    exterior: tuple[tuple[float, float], ...]
    cells: frozenset[Point]


@dataclass(frozen=True)
class Scenario:
    name: str
    width: int
    height: int
    layers: tuple[str, ...]
    groups: tuple[PowerGroup, ...]
    keepouts: dict[str, frozenset[Point]]
    obstacles: dict[str, tuple[KeepoutShape, ...]] | None = None
    clearance: int = 1
    min_width: int = 3

    @property
    def pins(self) -> tuple[Pin, ...]:
        return tuple(pin for group in self.groups for pin in group.pins)


@dataclass(frozen=True)
class Layout:
    layers: dict[str, dict[Point, str]]
    group_cells: dict[str, frozenset[Point]]
    group_polygons: dict[str, tuple["CopperPolygon", ...]]


@dataclass(frozen=True)
class ValidationReport:
    errors: list[str]


@dataclass(frozen=True)
class CopperPolygon:
    group_id: str
    layer: str
    exterior: tuple[tuple[float, float], ...]
    holes: tuple[tuple[tuple[float, float], ...], ...]
    area: float
