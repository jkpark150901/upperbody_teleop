"""Process 2 of 3: mocap (Axis Studio/MocapApi) body+arm retargeting.

Connects to MocapApi, runs RBY1UpperBodyRetargeter (arm/torso/head IK),
shows a live skeleton graph (x,y,z + rx,ry,rz per BVH landmark, see
scripts/skeleton_graph.py) and optionally records raw landmarks to a
.jsonl file -- but does NOT touch MuJoCo at all. Retargeted joint angles
go out over UDP (teleop/articulation_protocol.py's ArticulationSender) to
scripts/mujoco_physics_process.py, which is the only thing that steps
physics.

Run all three processes with run_all.bat, or standalone:
    python -m scripts.mocap_process --port 7002 --send udp://127.0.0.1:6100
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

_ROOT = pathlib.Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from robot_hand.hand_retarget import Dg5fHandRetargeter  # noqa: E402
from robot_hand.upper_body_retarget import RBY1UpperBodyRetargeter  # noqa: E402
from robot_hand.urdf_fk import load_urdf  # noqa: E402
from scripts.skeleton_graph import SkeletonGraphWindow  # noqa: E402
from scripts.upper_body_retarget_vedo import (  # noqa: E402
    CombinedRecorder, CombinedReplayReader, MocapLandmarkReader, ReplayLandmarkReader, _URDF,
    _HAND_JOINT_PREFIX, _HAND_URDF_PATHS, _hand_flexion, _hand_spread, _replay_format,
)
from teleop.articulation_protocol import ArticulationSender  # noqa: E402

_LANDMARK_KEYS = (
    "hips", "chest", "head", "left_shoulder", "left_elbow", "left_wrist",
    "right_shoulder", "right_elbow", "right_wrist",
)
_EDGES = (
    ("chest", "hips"), ("chest", "head"),
    ("chest", "left_shoulder"), ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("chest", "right_shoulder"), ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
)
_ROTATION_KEYS = {"head": "head_rotation", "left_wrist": "left_wrist_rotation", "right_wrist": "right_wrist_rotation"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("mocapapi", "replay"), default="mocapapi")
    parser.add_argument("--replay-file", type=pathlib.Path, default=None,
                         help="--backend replay: a --record'd .jsonl (e.g. recordings/session1.jsonl)")
    parser.add_argument("--no-loop", action="store_true", help="--backend replay: play once instead of looping")
    parser.add_argument("--port", type=int, default=7002, help="MocapApi UDP port")
    parser.add_argument("--send", default="udp://127.0.0.1:6100",
                         help="scripts/mujoco_physics_process.py's --port, as udp://host:port")
    parser.add_argument("--calibration-pose", choices=("attention", "tpose", "raise"), default="attention")
    parser.add_argument("--torso", choices=("fixed", "retarget"), default="fixed")
    parser.add_argument(
        "--arm-solver", choices=("closed-form", "mink"), default="closed-form",
        help="closed-form: RBY1UpperBodyRetargeter's own ZXZ/ZYZ branch-search IK. "
        "mink: solve elbow+wrist as a mink QP task instead (requires `pip install mink`).",
    )
    parser.add_argument("--bvh-scale", action="store_true",
                         help="scale each robot arm segment by how extended the human's own is")
    parser.add_argument("--record", type=pathlib.Path, default=None, help="record raw landmarks to this .jsonl file")
    parser.add_argument("--hz", type=float, default=30.0)
    args = parser.parse_args()
    if args.backend == "replay" and args.replay_file is None:
        parser.error("--backend replay requires --replay-file")

    tree = load_urdf(str(_URDF))
    retargeter = RBY1UpperBodyRetargeter(tree, calibration_pose=args.calibration_pose)
    retargeter.use_bvh_scale = args.bvh_scale
    mink_ik = None
    if args.arm_solver == "mink":
        from robot_hand.mink_upper_body_retarget import MinkUpperBodyIK  # optional dep, see that module
        mink_ik = MinkUpperBodyIK(retargeter, _URDF)

    if args.backend == "replay":
        replay_kind = _replay_format(args.replay_file)
        reader = (
            CombinedReplayReader(args.replay_file, loop=not args.no_loop) if replay_kind == "combined"
            else ReplayLandmarkReader(args.replay_file, loop=not args.no_loop)
        )
    else:
        reader = MocapLandmarkReader(args.port)
    reader.connect()
    sender = ArticulationSender(args.send, retargeter.names)

    # Hand flexion, when the reader has any (MocapLandmarkReader's own BVH
    # finger tracking live, or a recorded session's SenseGlove/Quest
    # flexion on replay -- both expose it the same way, via .last_hand,
    # see MocapLandmarkReader._bvh_hand_state / CombinedReplayReader). A
    # second ArticulationSender, same --send endpoint as the arm one:
    # ArticulationReceiver already merges frames from multiple senders by
    # their own layout id (see scripts/hand_process.py, which does exactly
    # this for Quest), so this doesn't collide with it.
    hand_retargeters = {
        side: Dg5fHandRetargeter(load_urdf(str(_HAND_URDF_PATHS[side])), prefix)
        for side, prefix in _HAND_JOINT_PREFIX.items()
    }
    hand_joint_names = [
        name for r in hand_retargeters.values() for names in r.chains.values() for name in names
    ]
    hand_sender = ArticulationSender(args.send, hand_joint_names)

    recorder = CombinedRecorder(args.record) if args.record else None
    graph = SkeletonGraphWindow("Mocap skeleton (x,y,z / rx,ry,rz)", _LANDMARK_KEYS, _EDGES)

    source_desc = f"replay {args.replay_file}" if args.backend == "replay" else f"MocapApi UDP :{args.port}"
    print(f"[mocap_process] {source_desc} -> {args.send}"
          + (f", recording to {args.record}" if args.record else ""))

    calibrated = False
    tick_interval = 1.0 / args.hz
    last_tick = 0.0
    try:
        while True:
            now = time.monotonic()
            if now - last_tick < tick_interval:
                time.sleep(max(0.0, tick_interval - (now - last_tick)))
                continue
            last_tick = now

            frame = reader.poll()
            if frame is None:
                continue
            if not calibrated:
                retargeter.calibrate(frame)
                calibrated = True
                print(f"[mocap_process] calibrated ({args.calibration_pose})")

            joints = retargeter.update(frame, retarget_torso=args.torso == "retarget")
            if mink_ik is not None:
                joints.update(mink_ik.update(frame))
            sender.send(
                [joints[n] for n in retargeter.names], time.time(), {"left": False, "right": False}
            )

            last_hand = getattr(reader, "last_hand", None) or {}
            hand_joints: dict = {}
            hand_present = {"left": False, "right": False}
            for side, sample in last_hand.items():
                flexion = _hand_flexion(sample)
                if flexion is None:
                    continue
                hand_present[side] = True
                hand_joints.update(hand_retargeters[side].retarget(flexion, _hand_spread(sample)))
            if any(hand_present.values()):
                hand_sender.send(
                    [hand_joints.get(n, 0.0) for n in hand_joint_names], time.time(), hand_present
                )

            if recorder is not None:
                recorder.write_frame(frame, {"left": None, "right": None})

            positions = {key: getattr(frame, key) for key in _LANDMARK_KEYS}
            rotations = {
                key: getattr(frame, attr) for key, attr in _ROTATION_KEYS.items()
                if getattr(frame, attr) is not None
            }
            graph.update(positions, rotations)
    except KeyboardInterrupt:
        pass
    finally:
        reader.close()
        sender.close()
        hand_sender.close()
        if recorder is not None:
            recorder.close()
        graph.close()


if __name__ == "__main__":
    main()
