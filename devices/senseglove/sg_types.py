"""Data types produced by the SenseGlove reader."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np

FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
N_FINGERS = 5
N_JOINTS_PER_FINGER = 4  # e.g. CMC/MCP, PIP, DIP, fingertip


@dataclass
class HandState:
    """One synchronized SenseGlove sample for a single hand.

    flexion         : (5,) normalized 0 (open) .. 1 (fully closed) per finger,
                       order = FINGER_NAMES. This is the "Phase 1 simple
                       mapping" quantity from the plan (section 9).
    joint_positions : (5, 4, 3) per-finger joint positions in the *wrist-local*
                       frame (meters), used only for point-level visualization
                       here (no retargeting math applied to it). If the real
                       glove doesn't expose forward-kinematics joint
                       positions, this is left as an empty array and the
                       viewer falls back to fingertip-only estimates.
    raw             : vendor-native raw payload (sensor angles etc.), kept
                       for recording per plan section 12 ("raw teleoperation
                       input을 반드시 별도로 저장한다").
    """

    timestamp: float
    hand: str  # "left" | "right"
    flexion: np.ndarray = field(default_factory=lambda: np.zeros(N_FINGERS))
    joint_positions: np.ndarray = field(
        default_factory=lambda: np.zeros((N_FINGERS, N_JOINTS_PER_FINGER, 3))
    )
    raw: Dict[str, Any] = field(default_factory=dict)
