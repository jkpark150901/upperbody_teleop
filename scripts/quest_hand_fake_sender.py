"""Stand-in for the Quest 3 XrHandsFB APK: sends synthetic hands over UDP.

Lets the receiving side (devices/quest_hand, `--hand-backend quest`) be run
and checked before the modified APK is built and adb-installed. Speaks the
exact wire format XrHandsFB's SendHandJointsUDP sends (see
devices/quest_hand/quest_hand_reader.py's module docstring): one 752-byte
binary packet per hand per tick over UDP, fingers sweeping
open -> closed -> open (the right hand is phase-shifted so the two sides are
distinguishable).

Usage:
    python scripts/quest_hand_fake_sender.py [--host 127.0.0.1] [--port 5005] [--hz 60]
    python scripts/upper_body_retarget_vedo.py --backend mock --hand-backend quest
"""

from __future__ import annotations

import argparse
import math
import pathlib
import socket
import struct
import sys
import time

_ROOT = pathlib.Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402

from devices.quest_hand.quest_hand_reader import (  # noqa: E402
    HEADER_SIZE,
    MAGIC,
    N_XR_JOINTS,
    PACKET_SIZE,
    _HEADER_FMT,
)
from devices.quest_hand.quest_hand_synth import synthetic_xr_hand  # noqa: E402


def encode_packet(hand: str, seq: int, t: float, joints, tracked_mask: int) -> bytes:
    header = struct.pack(
        _HEADER_FMT,
        MAGIC,
        1,  # version
        0 if hand == "left" else 1,
        1,  # active
        0,  # reserved
        seq,
        t,
        tracked_mask,
    )
    body = np.asarray(joints, dtype="<f4").tobytes()
    packet = header + body
    assert len(packet) == PACKET_SIZE, (len(packet), PACKET_SIZE, HEADER_SIZE)
    return packet


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", default="127.0.0.1", help="quest_hand_reader's --quest-host")
    ap.add_argument("--port", type=int, default=5005)
    ap.add_argument("--hz", type=float, default=60.0)
    ap.add_argument("--period", type=float, default=4.0, help="seconds per open/close cycle")
    args = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"[fake_sender] sending to {args.host}:{args.port}")

    # Hands sit at arbitrary world poses to exercise the wrist-local math.
    poses = {
        "left": (np.eye(3), np.array([-0.2, 1.1, -0.4])),
        "right": (np.eye(3), np.array([0.2, 1.1, -0.4])),
    }
    all_tracked = (1 << N_XR_JOINTS) - 1
    seq = {"left": 0, "right": 0}
    t0 = time.time()
    try:
        while True:
            t = time.time() - t0
            for side, phase in (("left", 0.0), ("right", 0.5)):
                curl = 0.5 - 0.5 * math.cos(2 * math.pi * (t / args.period + phase))
                rot, pos = poses[side]
                packet = encode_packet(
                    side, seq[side], t, synthetic_xr_hand(curl, rot, pos), all_tracked
                )
                sock.sendto(packet, (args.host, args.port))
                seq[side] += 1
            time.sleep(1.0 / args.hz)
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()


if __name__ == "__main__":
    main()
