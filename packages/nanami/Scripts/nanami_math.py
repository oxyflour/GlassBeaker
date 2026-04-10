from __future__ import annotations

import math


def identity():
    return [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]


def mat_mul(a, b):
    out = [[0.0] * 4 for _ in range(4)]
    for row in range(4):
        for col in range(4):
            out[row][col] = sum(a[row][i] * b[i][col] for i in range(4))
    return out


def translate(x: float, y: float, z: float):
    out = identity()
    out[0][3], out[1][3], out[2][3] = x, y, z
    return out


def rotate_rpy(roll: float, pitch: float, yaw: float):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr, 0.0],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr, 0.0],
        [-sp, cp * sr, cp * cr, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def rotate_axis(axis_xyz: list[float], angle: float):
    x, y, z = axis_xyz
    mag = math.sqrt(x * x + y * y + z * z) or 1.0
    x, y, z = x / mag, y / mag, z / mag
    c, s, t = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    return [
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y, 0.0],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x, 0.0],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def motion_matrix(joint: dict, value: float):
    if joint["type"] == "prismatic":
        x, y, z = joint["axis_xyz"]
        return translate(x * value, y * value, z * value)
    if joint["type"] in {"revolute", "continuous"}:
        return rotate_axis(joint["axis_xyz"], value)
    return identity()


def joint_matrix(joint: dict, value: float):
    origin = mat_mul(
        translate(*joint["origin_xyz"]),
        rotate_rpy(*joint["origin_rpy"]),
    )
    return mat_mul(origin, motion_matrix(joint, value))


def matrix_to_pose(matrix):
    pitch = math.asin(-max(-1.0, min(1.0, matrix[2][0])))
    if abs(math.cos(pitch)) < 1e-6:
        roll = 0.0
        yaw = math.atan2(-matrix[0][1], matrix[1][1])
    else:
        roll = math.atan2(matrix[2][1], matrix[2][2])
        yaw = math.atan2(matrix[1][0], matrix[0][0])
    location = [matrix[0][3], matrix[1][3], matrix[2][3]]
    rotation_deg = [math.degrees(pitch), math.degrees(yaw), math.degrees(roll)]
    return location, rotation_deg
