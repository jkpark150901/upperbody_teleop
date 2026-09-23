"""No-hardware check of the Quest hand-tracking UDP receive path.

1. Wire format: encode_packet() -> decode_packet() round-trips exactly.
2. Pure math: synthetic hands at several world poses must give the same
   flexion (pose-independent), monotonic in curl, and wrist-local joint
   positions that don't move with the world pose.
3. End to end: QuestHandUDPReader against scripts/quest_hand_fake_sender.py
   over real UDP sockets (sender started as a subprocess).

Usage:  python scripts/quest_hand_selfcheck.py
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import time

_ROOT = pathlib.Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402

from devices.quest_hand.quest_hand_reader import (  # noqa: E402
    N_XR_JOINTS,
    PACKET_SIZE,
    QuestHandUDPReader,
    decode_packet,
    xr_hand_to_state,
)
from devices.quest_hand.quest_hand_synth import synthetic_xr_hand  # noqa: E402
from scripts.quest_hand_fake_sender import encode_packet  # noqa: E402

_PORT = 19860


def _rot(axis: str, a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return {
        "x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
        "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
        "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]),
    }[axis]


def check_wire_format() -> None:
    joints = synthetic_xr_hand(0.6)
    all_tracked = (1 << N_XR_JOINTS) - 1
    packet = encode_packet("right", 42, 12.5, joints, all_tracked)
    assert len(packet) == PACKET_SIZE
    d = decode_packet(packet)
    assert d is not None
    assert d["hand"] == "right" and d["seq"] == 42 and d["active"] is True
    assert abs(d["t"] - 12.5) < 1e-9
    assert d["tracked_mask"] == all_tracked
    np.testing.assert_allclose(d["joints"], np.asarray(joints, dtype="<f4"), atol=1e-6)
    assert decode_packet(b"short") is None
    assert decode_packet(b"\x00" * PACKET_SIZE) is None  # bad magic
    print("wire format OK")


def check_math() -> None:
    poses = [
        (np.eye(3), np.zeros(3)),
        (_rot("y", 1.1) @ _rot("x", -0.7), np.array([0.3, 1.2, -0.5])),
        (_rot("z", 2.9) @ _rot("y", 0.4), np.array([-1.0, 0.2, 0.8])),
    ]
    prev = None
    ref = None
    for curl in (0.0, 0.25, 0.5, 0.75, 1.0):
        states = [xr_hand_to_state("left", synthetic_xr_hand(curl, r, p)) for r, p in poses]
        for s in states[1:]:
            np.testing.assert_allclose(s.flexion, states[0].flexion, atol=1e-6)
            np.testing.assert_allclose(s.joint_positions, states[0].joint_positions, atol=1e-6)
        f = states[0].flexion
        if prev is not None:
            assert np.all(f >= prev - 1e-9), f"flexion not monotonic at curl={curl}: {f} < {prev}"
        prev = f
        print(f"  curl={curl:4.2f} flexion={np.round(f, 2)}")
        ref = f
    assert np.all(np.isclose(xr_hand_to_state("left", synthetic_xr_hand(0.0)).flexion, 0, atol=0.05))
    assert np.all(ref > 0.95), f"fully curled hand should read ~1, got {ref}"

    # untracked joints -> dropped
    tracked = [True] * N_XR_JOINTS
    tracked[8] = False
    assert xr_hand_to_state("left", synthetic_xr_hand(0.5), tracked) is None
    print("math OK")


def check_e2e() -> None:
    sender = subprocess.Popen(
        [sys.executable, str(_ROOT / "scripts" / "quest_hand_fake_sender.py"),
         "--host", "127.0.0.1", "--port", str(_PORT), "--period", "2"],
        stdout=subprocess.DEVNULL,
    )
    try:
        reader = QuestHandUDPReader("127.0.0.1", _PORT)
        reader.connect()
        seen = {"left": [], "right": []}
        t_end = time.time() + 2.5
        while time.time() < t_end:
            left, right = reader.poll()
            for name, hs in (("left", left), ("right", right)):
                if hs is not None:
                    assert hs.hand == name and hs.flexion.shape == (5,)
                    assert hs.joint_positions.shape == (5, 4, 3)
                    seen[name].append(float(hs.flexion.mean()))
        reader.close()
        for name, vals in seen.items():
            assert len(vals) > 20, f"{name}: only {len(vals)} samples"
            assert max(vals) - min(vals) > 0.8, f"{name}: flexion didn't sweep ({min(vals):.2f}..{max(vals):.2f})"
            print(f"  {name}: {len(vals)} samples, flexion {min(vals):.2f}..{max(vals):.2f}")
        print("end-to-end OK")
    finally:
        sender.terminate()


if __name__ == "__main__":
    check_wire_format()
    check_math()
    check_e2e()
