"""Data types produced by the Perception Neuron reader."""

from __future__ import annotations

from dataclasses import dataclass

from teleop.geometry import Pose


@dataclass
class PNFrame:
    """One synchronized Perception Neuron sample.

    All poses are expressed in the Axis Studio / MocapApi world frame
    (whatever that happens to be — global position + orientation, not yet
    torso-relative). Torso-relative conversion happens downstream in
    teleop/calibration.py so the raw stream can always be recorded as-is
    (see plan section 12: "Raw teleoperation input을 반드시 별도로 저장한다").
    """

    timestamp: float
    chest: Pose
    left_wrist: Pose
    right_wrist: Pose
