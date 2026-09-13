"""Minimal SE(3) pose utilities shared across devices/teleop/mujoco_teleop.

Quaternion convention: (w, x, y, z), matching MuJoCo's mjData/mjModel convention.
Device-specific readers are responsible for converting their native quaternion
order (e.g. MocapApi/Unity-style x,y,z,w) into this convention at the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """wxyz quaternion -> 3x3 rotation matrix."""
    w, x, y, z = q
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array([
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)],
    ])


def mat_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> wxyz quaternion."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def quat_from_xyzw(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Convert Unity/MocapApi-style xyzw quaternion into internal wxyz."""
    return np.array([w, x, y, z])


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_conj(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def quat_identity() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0, 0.0])


def euler_xyz_to_quat(euler_deg: np.ndarray) -> np.ndarray:
    """Extrinsic XYZ Euler angles (degrees) -> wxyz quaternion."""
    rx, ry, rz = np.deg2rad(np.asarray(euler_deg, dtype=float))

    def axis_quat(angle, axis):
        half = angle / 2.0
        s = np.sin(half)
        q = np.array([np.cos(half), 0.0, 0.0, 0.0])
        q[1 + axis] = s
        return q

    qx = axis_quat(rx, 0)
    qy = axis_quat(ry, 1)
    qz = axis_quat(rz, 2)
    # extrinsic X then Y then Z == intrinsic composition qz * qy * qx
    return quat_mul(quat_mul(qz, qy), qx)


@dataclass
class Pose:
    """Rigid transform: position (3,) + wxyz quaternion (4,)."""

    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    quat: np.ndarray = field(default_factory=quat_identity)

    def __post_init__(self):
        self.pos = np.asarray(self.pos, dtype=float).reshape(3)
        self.quat = np.asarray(self.quat, dtype=float).reshape(4)

    @staticmethod
    def identity() -> "Pose":
        return Pose(np.zeros(3), quat_identity())

    def as_matrix(self) -> np.ndarray:
        return quat_to_mat(self.quat)

    def inverse(self) -> "Pose":
        R = self.as_matrix()
        Rt = R.T
        return Pose(-Rt @ self.pos, mat_to_quat(Rt))

    def compose(self, other: "Pose") -> "Pose":
        """self ∘ other: apply `other` first, then `self`."""
        R = self.as_matrix()
        return Pose(R @ other.pos + self.pos, quat_mul(self.quat, other.quat))

    def apply_to_point(self, p: np.ndarray) -> np.ndarray:
        return self.as_matrix() @ np.asarray(p, dtype=float) + self.pos

    def axis_endpoints(self, length: float = 0.05) -> np.ndarray:
        """Return world-space endpoints of the local +X/+Y/+Z axes, shape (3, 3)."""
        R = self.as_matrix()
        return np.stack([self.pos + R[:, i] * length for i in range(3)], axis=0)

    def copy(self) -> "Pose":
        return Pose(self.pos.copy(), self.quat.copy())
