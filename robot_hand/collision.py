"""Capsule-capsule geometry for DG5F finger self-collision.

Two uses:
- scripts/extract_hand_urdf.py calls `direction_rpy` to *author* a
  Z-aligned capsule's <origin rpy="..."> so it points along a given finger
  segment's bone direction when generating robot_hand/urdf/dg5f_*.urdf.
- robot_hand/hand_retarget.py calls `capsule_world_segment` +
  `segment_segment_distance` at retarget time to check whether adjacent
  fingers' capsules (as parsed back out of that URDF by
  robot_hand/urdf_fk.py) would interpenetrate.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from teleop.geometry import Pose


def direction_rpy(direction: np.ndarray) -> np.ndarray:
    """roll/pitch/yaw (URDF fixed-axis convention, radians) that rotates the
    local +Z axis to point along `direction`. Roll is left at 0 -- a
    capsule is rotationally symmetric about its own axis, so it doesn't
    matter here.
    """
    d = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(d)
    if norm < 1e-9:
        return np.zeros(3)
    dx, dy, dz = d / norm
    r_xy = float(np.hypot(dx, dy))
    pitch = float(np.arctan2(r_xy, dz))
    yaw = float(np.arctan2(dy, dx)) if r_xy > 1e-9 else 0.0
    return np.array([0.0, pitch, yaw])


def capsule_world_segment(link_pose: Pose, capsule_origin: Pose, length: float) -> Tuple[np.ndarray, np.ndarray]:
    """World-space endpoints of a capsule's central segment.

    `capsule_origin` is the capsule's own <collision><origin> pose, in the
    owning link's local frame, with the capsule spanning +-length/2 along
    its local +Z (see direction_rpy / extract_hand_urdf.py).
    """
    world = link_pose.compose(capsule_origin)
    R = world.as_matrix()
    half = R[:, 2] * (length / 2.0)
    return world.pos - half, world.pos + half


def segment_segment_distance(p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray) -> float:
    """Shortest distance between segments p1-q1 and p2-q2 (Ericson, "Real-
    Time Collision Detection", closest-point-between-segments)."""
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = float(np.dot(d1, d1))
    e = float(np.dot(d2, d2))
    f = float(np.dot(d2, r))

    eps = 1e-12
    if a <= eps and e <= eps:
        return float(np.linalg.norm(p1 - p2))
    if a <= eps:
        t = np.clip(f / e, 0.0, 1.0)
        s = 0.0
    else:
        c = float(np.dot(d1, r))
        if e <= eps:
            t = 0.0
            s = np.clip(-c / a, 0.0, 1.0)
        else:
            b = float(np.dot(d1, d2))
            denom = a * e - b * b
            s = np.clip((b * f - c * e) / denom, 0.0, 1.0) if denom > eps else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = np.clip(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t = 1.0
                s = np.clip((b - c) / a, 0.0, 1.0)

    closest1 = p1 + d1 * s
    closest2 = p2 + d2 * t
    return float(np.linalg.norm(closest1 - closest2))
