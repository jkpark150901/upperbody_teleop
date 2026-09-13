"""JSON-over-UDP wire format between the device PC (PN + SenseGlove) and the
MuJoCo PC.

JSON was chosen over a packed struct because payloads are small (well under
typical MTU at the target 50-100 Hz control rate — see plan section 6/14)
and human-readable packets make the link trivial to debug with e.g.
`nc -ul <port>` while wiring up two separate machines. `raw` device payloads
are intentionally NOT included here to keep packets small and dependency-free
for the network hop; if/when a recorder needs the full raw stream (plan
section 12), record it on the sender side directly from the readers instead
of round-tripping it over UDP.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import numpy as np

from devices.senseglove.sg_types import HandState
from teleop.geometry import Pose
from teleop.teleop_state import HumanTeleopState

PROTOCOL_VERSION = 1


def _pose_to_dict(p: Pose) -> dict:
    return {"pos": p.pos.tolist(), "quat": p.quat.tolist()}


def _pose_from_dict(d: dict) -> Pose:
    return Pose(np.array(d["pos"], dtype=float), np.array(d["quat"], dtype=float))


def _hand_to_dict(h: HandState) -> dict:
    return {
        "hand": h.hand,
        "flexion": h.flexion.tolist(),
        "joint_positions": h.joint_positions.tolist(),
    }


def _hand_from_dict(d: dict) -> HandState:
    return HandState(
        timestamp=0.0,
        hand=d["hand"],
        flexion=np.array(d["flexion"], dtype=float),
        joint_positions=np.array(d["joint_positions"], dtype=float),
    )


def encode_state(state: HumanTeleopState, seq: int) -> bytes:
    payload = {
        "v": PROTOCOL_VERSION,
        "seq": seq,
        "t_send": time.time(),
        "t_sample": state.timestamp,
        "chest": _pose_to_dict(state.chest),
        "left_wrist": _pose_to_dict(state.left_wrist),
        "right_wrist": _pose_to_dict(state.right_wrist),
        "left_hand": _hand_to_dict(state.left_hand),
        "right_hand": _hand_to_dict(state.right_hand),
    }
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


@dataclass
class ReceivedTeleopPacket:
    seq: int
    t_send: float
    t_sample: float
    t_recv: float
    chest: Pose
    left_wrist: Pose
    right_wrist: Pose
    left_hand: HandState
    right_hand: HandState


def decode_state(data: bytes) -> ReceivedTeleopPacket:
    d = json.loads(data.decode("utf-8"))
    if d.get("v") != PROTOCOL_VERSION:
        raise ValueError(f"unsupported protocol version: {d.get('v')}")
    return ReceivedTeleopPacket(
        seq=d["seq"],
        t_send=d["t_send"],
        t_sample=d["t_sample"],
        t_recv=time.time(),
        chest=_pose_from_dict(d["chest"]),
        left_wrist=_pose_from_dict(d["left_wrist"]),
        right_wrist=_pose_from_dict(d["right_wrist"]),
        left_hand=_hand_from_dict(d["left_hand"]),
        right_hand=_hand_from_dict(d["right_hand"]),
    )
