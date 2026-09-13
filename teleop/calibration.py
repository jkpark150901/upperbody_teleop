"""Recenter + wrist retargeting math (plan section 7).

Runs on the MuJoCo-side process: it owns the notion of "robot workspace",
so recentering and workspace clamping naturally live where the operator is
watching the viewer, not on the device PC.

Position mapping (plan 7.2):
    p_target = p_robot0 + S * R_align * (p_human - p_human0)

Orientation mapping (plan 7.3):
    R_target = R_robot0 * R_human0^-1 * R_human * R_offset

`p_human` / `R_human` here are the *torso-relative* wrist pose
(T_chest^-1 * T_wrist, plan section 3.1) rather than raw global PN pose, to
reject PN IMU drift and decouple human walking from robot base motion.
`p_robot0` / `R_robot0` are simplified to a single configured "marker
origin" pose per side, since there's no real robot/URDF yet — once one
exists, swap `marker_origin` for the actual robot EE pose at calibration
time.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from teleop.geometry import Pose, quat_conj, quat_mul


@dataclass
class SideCalibration:
    """Per-arm (left/right) calibration reference + retargeting config."""

    marker_origin: Pose  # "robot0": nominal target pose at calibration time
    scale: float = 0.7  # S
    align_quat: np.ndarray = None  # R_align: human -> robot/world frame axes
    offset_quat: np.ndarray = None  # R_offset: human neutral vs robot neutral wrist orientation

    clamp_min: np.ndarray = None
    clamp_max: np.ndarray = None
    max_linear_velocity: float = 1.5  # m/s

    def __post_init__(self):
        if self.align_quat is None:
            self.align_quat = np.array([1.0, 0.0, 0.0, 0.0])
        if self.offset_quat is None:
            self.offset_quat = np.array([1.0, 0.0, 0.0, 0.0])
        if self.clamp_min is None:
            self.clamp_min = np.array([-1.0, -1.0, 0.0])
        if self.clamp_max is None:
            self.clamp_max = np.array([1.0, 1.0, 2.0])


class WristRetargeter:
    """Holds calibration state for both arms and computes retargeted targets."""

    def __init__(self, left_cfg: SideCalibration, right_cfg: SideCalibration):
        self._cfg = {"left": left_cfg, "right": right_cfg}
        self._ref_rel: dict[str, Optional[Pose]] = {"left": None, "right": None}
        self._prev_target: dict[str, Optional[Pose]] = {"left": None, "right": None}
        self._prev_t: dict[str, Optional[float]] = {"left": None, "right": None}
        self.calibrated = False

    def recenter(self, chest: Pose, left_wrist: Pose, right_wrist: Pose) -> None:
        chest_inv = chest.inverse()
        self._ref_rel["left"] = chest_inv.compose(left_wrist)
        self._ref_rel["right"] = chest_inv.compose(right_wrist)
        self._prev_target["left"] = None
        self._prev_target["right"] = None
        self.calibrated = True
        print("[calibration] recentered")

    def update(self, side: str, chest: Pose, wrist: Pose) -> Optional[Pose]:
        """Compute the retargeted target pose for one arm. None if not yet calibrated."""
        ref = self._ref_rel[side]
        if ref is None:
            return None
        cfg = self._cfg[side]

        rel = chest.inverse().compose(wrist)

        delta_p = rel.pos - ref.pos
        R_align = _quat_to_mat_local(cfg.align_quat)
        target_pos = cfg.marker_origin.pos + cfg.scale * (R_align @ delta_p)
        target_pos = np.clip(target_pos, cfg.clamp_min, cfg.clamp_max)

        target_quat = quat_mul(
            quat_mul(quat_mul(cfg.align_quat, quat_conj(ref.quat)), rel.quat),
            cfg.offset_quat,
        )

        now = time.time()
        prev = self._prev_target[side]
        prev_t = self._prev_t[side]
        if prev is not None and prev_t is not None:
            dt = max(now - prev_t, 1e-4)
            max_step = cfg.max_linear_velocity * dt
            delta = target_pos - prev.pos
            dist = np.linalg.norm(delta)
            if dist > max_step and dist > 1e-9:
                target_pos = prev.pos + delta * (max_step / dist)

        target = Pose(target_pos, target_quat)
        self._prev_target[side] = target
        self._prev_t[side] = now
        return target


def _quat_to_mat_local(q: np.ndarray) -> np.ndarray:
    from teleop.geometry import quat_to_mat
    return quat_to_mat(q)


def map_hand_points_to_world(wrist_target: Pose, joint_positions_local: np.ndarray) -> np.ndarray:
    """Rigidly attach glove-local finger joint points to a retargeted wrist pose.

    joint_positions_local: (5, 4, 3) in the wrist-local frame (see HandState).
    Returns the same shape in world frame.
    """
    R = wrist_target.as_matrix()
    return joint_positions_local @ R.T + wrist_target.pos
