"""Synthetic 26-joint XR hand for testing the Quest UDP path without a
headset or a built APK.

Builds a planar-curl hand in OpenXR joint order and places it at an
arbitrary world pose, so consumers can check that flexion/local-frame math
doesn't depend on where the hand is. Used by scripts/quest_hand_fake_sender.py
and the self-check in scripts/quest_hand_selfcheck.py.
"""

from __future__ import annotations

from typing import List

import numpy as np

from devices.quest_hand.quest_hand_reader import _CHAIN, N_XR_JOINTS, XR_PALM, XR_WRIST
from devices.senseglove.sg_types import FINGER_NAMES

# Finger base offset in the wrist frame (x = across the palm, z = -forward)
# and bone lengths (m), base -> tip.
_BASE = {
    "thumb": (-0.03, 0.0, -0.02),
    "index": (-0.03, 0.0, -0.09),
    "middle": (-0.01, 0.0, -0.095),
    "ring": (0.01, 0.0, -0.09),
    "pinky": (0.03, 0.0, -0.08),
}
_BONES = {
    "thumb": (0.04, 0.03, 0.025),
    "index": (0.04, 0.025, 0.02, 0.015),
    "middle": (0.045, 0.028, 0.022, 0.015),
    "ring": (0.04, 0.025, 0.02, 0.015),
    "pinky": (0.032, 0.02, 0.016, 0.014),
}
# Total curl (rad) at curl=1.0; matches the reader's default closed range.
_TOTAL_CURL = {"thumb": 1.6, "index": 4.0, "middle": 4.0, "ring": 4.0, "pinky": 4.0}


def _rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _matrix_to_quat_xyzw(r: np.ndarray) -> np.ndarray:
    w = np.sqrt(max(0.0, 1 + r[0, 0] + r[1, 1] + r[2, 2])) / 2
    if w > 1e-6:
        return np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1], 4 * w * w]) / (4 * w)
    # 180 deg rotation: pick the largest diagonal axis
    i = int(np.argmax(np.diag(r)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(max(0.0, r[i, i] - r[j, j] - r[k, k] + 1)) * 2
    q = np.zeros(4)
    q[i] = s / 4
    q[j] = (r[j, i] + r[i, j]) / s
    q[k] = (r[k, i] + r[i, k]) / s
    q[3] = (r[k, j] - r[j, k]) / s
    return q


def synthetic_xr_hand(
    curl: float,
    world_rot: np.ndarray | None = None,
    world_pos: np.ndarray | None = None,
) -> List[List[float]]:
    """(26, 7) joint list; every finger curled by ``curl`` in 0..1."""
    r_w = np.eye(3) if world_rot is None else world_rot
    p_w = np.zeros(3) if world_pos is None else world_pos
    local = np.zeros((N_XR_JOINTS, 3))
    local[XR_WRIST] = 0.0
    local[XR_PALM] = (0.0, 0.0, -0.05)

    for name in FINGER_NAMES:
        chain = _CHAIN[name]
        bones = _BONES[name]
        # Non-thumb chains start at the metacarpal; give it a fixed, uncurled
        # length so the MCP angle is the first measured bend.
        p = np.array(_BASE[name], dtype=float)
        local[chain[0]] = p
        angle = 0.0
        n_bends = len(chain) - 2
        for i, length in enumerate(bones):
            if i > 0:
                angle += curl * _TOTAL_CURL[name] / n_bends
            p = p + _rot_x(angle) @ np.array([0.0, 0.0, -length])
            local[chain[i + 1]] = p

    world = local @ r_w.T + p_w
    quat = _matrix_to_quat_xyzw(r_w)
    out = np.zeros((N_XR_JOINTS, 7))
    out[:, :3] = world
    out[:, 3:] = quat
    return out.tolist()
