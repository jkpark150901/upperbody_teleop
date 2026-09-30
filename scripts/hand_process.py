"""Process 1 of 3: Quest hand-tracking retargeting.

Connects to the Quest 3 hand-tracking UDP stream, retargets each hand's 26
XR joints to DG5F finger joint angles, shows a live skeleton graph (x,y,z +
rx,ry,rz per XR joint, see scripts/skeleton_graph.py) and optionally
records the raw 26-joint stream to a .jsonl file -- but does NOT touch
MuJoCo at all. Retargeted joint angles go out over UDP (teleop/
articulation_protocol.py's ArticulationSender) to scripts/
mujoco_physics_process.py, which is the only thing that steps physics.

Run all three processes with run_all.bat, or standalone:
    python -m scripts.hand_process --quest-host 192.168.8.156 --quest-port 5005 --send udp://127.0.0.1:6100
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Dict, Optional

_ROOT = pathlib.Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402

from devices.quest_hand.quest_hand_reader import (  # noqa: E402
    N_XR_JOINTS, XR_WRIST, QuestHandUDPReader, _CHAIN, _quat_xyzw_to_matrix,
)
from devices.senseglove.sg_types import FINGER_NAMES  # noqa: E402
from robot_hand.anydex_retarget import AnyDexDg5fRetargeter  # noqa: E402
from robot_hand.hand_retarget import Dg5fHandRetargeter  # noqa: E402
from robot_hand.urdf_fk import load_urdf  # noqa: E402
from scripts.skeleton_graph import SkeletonGraphWindow  # noqa: E402
from scripts.upper_body_retarget_vedo import _HAND_JOINT_PREFIX, _HAND_URDF_PATHS  # noqa: E402
from teleop.articulation_protocol import ArticulationSender  # noqa: E402

_JOINT_NAMES = tuple(f"xr_{i}" for i in range(N_XR_JOINTS))
_EDGES = tuple(
    (f"xr_{a}", f"xr_{b}")
    for chain in _CHAIN.values()
    for a, b in zip((XR_WRIST,) + chain[:-1], chain)
)


class HandRecorder:
    """Raw 26-joint XR stream (position + quaternion, both hands) plus the
    retargeted joint angles that came from it, one .jsonl line per frame --
    unlike CombinedRecorder (mocap_process.py's), which only ever captures
    positions/rotation-matrices from BVH, this needs the RAW xr_joints
    array so a session can be replayed through a different hand retargeter
    later without re-capturing from the headset."""

    FORMAT = "hand_xr_v1"

    def __init__(self, path: pathlib.Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(path, "w", encoding="utf-8")
        self.t0: Optional[float] = None
        json.dump({"type": "header", "format": self.FORMAT}, self.file)
        self.file.write("\n")
        self.file.flush()

    def write_frame(self, xr_joints: Dict[str, Optional[np.ndarray]], retargeted: Dict[str, float]) -> None:
        now = time.monotonic()
        if self.t0 is None:
            self.t0 = now
        record = {
            "type": "frame",
            "t": now - self.t0,
            "xr_joints": {
                side: (None if j is None else j.tolist()) for side, j in xr_joints.items()
            },
            "retargeted": retargeted,
        }
        json.dump(record, self.file)
        self.file.write("\n")

    def close(self) -> None:
        self.file.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quest-host", default="192.168.8.156")
    parser.add_argument("--quest-port", type=int, default=5005)
    parser.add_argument("--send", default="udp://127.0.0.1:6100",
                         help="scripts/mujoco_physics_process.py's --port, as udp://host:port")
    parser.add_argument("--hand-retargeter", choices=("analytic", "anydex"), default="analytic")
    parser.add_argument("--anydex-config-left", type=pathlib.Path, default=None)
    parser.add_argument("--anydex-config-right", type=pathlib.Path, default=None)
    parser.add_argument("--record", type=pathlib.Path, default=None, help="record raw xr_joints to this .jsonl file")
    parser.add_argument("--hz", type=float, default=30.0)
    args = parser.parse_args()

    analytic_retargeters = {
        side: Dg5fHandRetargeter(load_urdf(str(_HAND_URDF_PATHS[side])), prefix)
        for side, prefix in _HAND_JOINT_PREFIX.items()
    }
    anydex_retargeters = {}
    if args.hand_retargeter == "anydex":
        config_paths = {"left": args.anydex_config_left, "right": args.anydex_config_right}
        anydex_retargeters = {side: AnyDexDg5fRetargeter(side, config_paths[side]) for side in ("left", "right")}

    all_names = [name for r in analytic_retargeters.values() for names in r.chains.values() for name in names]

    reader = QuestHandUDPReader(args.quest_host, args.quest_port)
    reader.connect()
    sender = ArticulationSender(args.send, all_names)
    recorder = HandRecorder(args.record) if args.record else None
    graph = SkeletonGraphWindow("Quest hand skeleton (x,y,z / rx,ry,rz)", _JOINT_NAMES, _EDGES)

    print(f"[hand_process] Quest UDP {args.quest_host}:{args.quest_port} -> {args.send}"
          + (f", recording to {args.record}" if args.record else ""))

    tick_interval = 1.0 / args.hz
    last_tick = 0.0
    last_hand: Dict[str, object] = {"left": None, "right": None}
    try:
        while True:
            now = time.monotonic()
            if now - last_tick < tick_interval:
                time.sleep(max(0.0, tick_interval - (now - last_tick)))
                continue
            last_tick = now

            left, right = reader.poll()
            if left is not None:
                last_hand["left"] = left
            if right is not None:
                last_hand["right"] = right

            joints: Dict[str, float] = {}
            xr_joints_by_side: Dict[str, Optional[np.ndarray]] = {"left": None, "right": None}
            positions: Dict[str, np.ndarray] = {}
            rotations: Dict[str, np.ndarray] = {}
            for side, sample in last_hand.items():
                if sample is None:
                    continue
                xr = sample.raw["xr_joints"]
                xr_joints_by_side[side] = xr
                if args.hand_retargeter == "anydex":
                    joints.update(anydex_retargeters[side].retarget_openxr_joints(xr))
                else:
                    joints.update(analytic_retargeters[side].retarget_openxr_joints(xr))
                for i in range(N_XR_JOINTS):
                    positions[f"xr_{i}"] = xr[i, :3]
                    rotations[f"xr_{i}"] = _quat_xyzw_to_matrix(xr[i, 3:])

            if joints:
                sender.send([joints.get(n, 0.0) for n in all_names], time.time(), {
                    "left": last_hand["left"] is not None, "right": last_hand["right"] is not None,
                })
            if recorder is not None:
                recorder.write_frame(xr_joints_by_side, joints)
            if positions:
                graph.update(positions, rotations)
    except KeyboardInterrupt:
        pass
    finally:
        reader.close()
        sender.close()
        if recorder is not None:
            recorder.close()
        graph.close()


if __name__ == "__main__":
    main()
