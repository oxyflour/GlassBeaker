from __future__ import annotations

import math

CAMERA_VERTICAL_APERTURE = 24.0


def focal_length_from_fovy(fovy: float, vertical_aperture: float = CAMERA_VERTICAL_APERTURE) -> float:
    radians = math.radians(float(fovy))
    return 0.5 * float(vertical_aperture) / math.tan(radians * 0.5)


def fovy_from_focal_length(focal_length: float, vertical_aperture: float) -> float:
    return math.degrees(2.0 * math.atan(float(vertical_aperture) * 0.5 / float(focal_length)))
