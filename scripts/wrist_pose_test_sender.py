"""Synthetic wrist-pose IK sender for MuJoCo debugging without VR.

Builds repeatable wrist position/orientation targets from the URDF forward
kinematics, solves them through RBY1UpperBodyRetargeter.solve_wrist_pose(),
and streams the resulting arm joint targets to scripts/mujoco_physics_process.py.

Example:
    python -m scripts.wrist_pose_test_sender --send udp://127.0.0.1:6100 --mode roll
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys
import time
from typing import Dict, Sequence

import numpy as np

_ROOT = pathlib.Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from robot_hand.upper_body_retarget import RBY1UpperBodyRetargeter  # noqa: E402
from robot_hand.urdf_fk import load_urdf  # noqa: E402
from scripts.hand_process import _ARM_NAMES, _WRIST_TARGET_NAMES, _WRIST_TARGET_ROT_NAMES  # noqa: E402
from scripts.skeleton_graph import SkeletonGraphWindow  # noqa: E402
from scripts.upper_body_retarget_mujoco import _quest_forward_reference  # noqa: E402
from teleop.articulation_protocol import ArticulationSender  # noqa: E402

_URDF = _ROOT / "rby1_dg5f.urdf"
_PALM_LINK = {"left": "left_hand_ll_dg_palm", "right": "right_hand_rl_dg_palm"}


def _rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


_AXIS_ROT = {"roll": _rot_x, "pitch": _rot_y, "yaw": _rot_z}


def _parse_sides(value: str) -> tuple[str, ...]:
    if value == "both":
        return ("left", "right")
    if value in ("left", "right"):
        return (value,)
    raise argparse.ArgumentTypeError("expected left, right, or both")


def _target_names() -> list[str]:
    return list(_ARM_NAMES) + list(_WRIST_TARGET_NAMES) + list(_WRIST_TARGET_ROT_NAMES)


def _base_reference(retargeter: RBY1UpperBodyRetargeter):
    base_wrist, base_rot = _quest_forward_reference(retargeter)
    base_values = retargeter._values(retargeter.q)
    return base_wrist, base_rot, {n: base_values[n] for n in _ARM_NAMES}


def _wrist_delta(mode: str, phase: float, amp_rad: float) -> np.ndarray:
    if mode in _AXIS_ROT:
        return _AXIS_ROT[mode](amp_rad * math.sin(phase))
    if mode == "combo":
        return (
            _rot_z(0.65 * amp_rad * math.sin(phase))
            @ _rot_y(0.50 * amp_rad * math.sin(phase + 1.7))
            @ _rot_x(amp_rad * math.sin(phase + 3.1))
        )
    raise ValueError(f"unknown mode: {mode}")


def _position_delta(mode: str, phase: float, amp_m: float) -> np.ndarray:
    if amp_m <= 0.0:
        return np.zeros(3)
    if mode == "circle":
        return amp_m * np.array([0.0, math.cos(phase), math.sin(phase)])
    return amp_m * np.array([0.35 * math.sin(phase * 0.5), math.sin(phase), 0.5 * math.sin(phase + 1.2)])


def _fill_target_values(out: Dict[str, float], side: str, wrist_pos: np.ndarray, wrist_rot: np.ndarray) -> None:
    for i, axis in enumerate(("x", "y", "z")):
        out[f"{side}_wrist_target_{axis}"] = float(wrist_pos[i])
    for row in range(3):
        for col in range(3):
            out[f"{side}_wrist_target_r{row}{col}"] = float(wrist_rot[row, col])


def _graph_state(retargeter: RBY1UpperBodyRetargeter, sides: Sequence[str], targets: Dict[str, np.ndarray]):
    links = []
    for side in sides:
        links += [f"link_{side}_arm_0", f"link_{side}_arm_3", f"link_{side}_arm_6", _PALM_LINK[side]]
    poses = retargeter.poses()
    positions: Dict[str, np.ndarray] = {}
    rotations: Dict[str, np.ndarray] = {}
    for side in sides:
        positions[f"{side}_shoulder"] = poses[f"link_{side}_arm_0"].pos
        positions[f"{side}_elbow"] = poses[f"link_{side}_arm_3"].pos
        positions[f"{side}_wrist"] = poses[f"link_{side}_arm_6"].pos
        positions[f"{side}_palm"] = poses[_PALM_LINK[side]].pos
        positions[f"{side}_target"] = targets[side]
        rotations[f"{side}_wrist"] = poses[f"link_{side}_arm_6"].as_matrix()
    return positions, rotations


def _graph_names_edges(sides: Sequence[str]):
    names = []
    edges = []
    for side in sides:
        side_names = [f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist", f"{side}_palm", f"{side}_target"]
        names.extend(side_names)
        edges.extend([
            (f"{side}_shoulder", f"{side}_elbow"),
            (f"{side}_elbow", f"{side}_wrist"),
            (f"{side}_wrist", f"{side}_palm"),
            (f"{side}_wrist", f"{side}_target"),
        ])
    return names, edges


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--send", default="udp://127.0.0.1:6100",
                        help="mujoco_physics_process.py endpoint")
    parser.add_argument("--hand", type=_parse_sides, default="both",
                        help="left, right, or both")
    parser.add_argument("--mode", choices=("roll", "pitch", "yaw", "combo"), default="roll",
                        help="orientation motion to replay")
    parser.add_argument("--hz", type=float, default=60.0)
    parser.add_argument("--period", type=float, default=4.0)
    parser.add_argument("--amplitude-deg", type=float, default=55.0,
                        help="orientation amplitude in degrees")
    parser.add_argument("--position-amplitude", type=float, default=0.0,
                        help="optional wrist-position wiggle amplitude in metres")
    parser.add_argument("--position-mode", choices=("lissajous", "circle"), default="lissajous")
    parser.add_argument("--graph-hz", type=float, default=15.0)
    parser.add_argument("--no-graph", action="store_true")
    parser.add_argument("--log-interval", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=0.0,
                        help="seconds to run before exiting; 0 means run until Ctrl+C")
    args = parser.parse_args()

    sides = args.hand
    tree = load_urdf(str(_URDF))
    retargeter = RBY1UpperBodyRetargeter(tree, calibration_pose="attention")
    base_wrist, base_rot, base_joints = _base_reference(retargeter)

    names = _target_names()
    sender = ArticulationSender(args.send, names)
    graph = None
    if not args.no_graph:
        graph_names, graph_edges = _graph_names_edges(sides)
        graph = SkeletonGraphWindow("Synthetic wrist pose IK test", graph_names, graph_edges)

    print(
        f"[wrist_pose_test_sender] sending to {args.send} hand={','.join(sides)} "
        f"mode={args.mode} amp={args.amplitude_deg:.1f}deg period={args.period:.2f}s"
    )
    print("[wrist_pose_test_sender] watch orient_err and clip(rad); nonzero clip means a wrist joint limit is active")

    tick = 1.0 / args.hz
    graph_tick = 1.0 / args.graph_hz
    amp_rad = math.radians(args.amplitude_deg)
    t0 = time.monotonic()
    last_send = 0.0
    last_graph = 0.0
    last_log = 0.0
    metrics: Dict[str, Dict[str, object]] = {}
    try:
        while True:
            now = time.monotonic()
            if now - last_send < tick:
                time.sleep(max(0.0, tick - (now - last_send)))
                continue
            last_send = now
            if args.duration > 0.0 and now - t0 >= args.duration:
                break
            phase = 2.0 * math.pi * ((now - t0) / args.period)

            joints: Dict[str, float] = dict(base_joints)
            targets_for_graph: Dict[str, np.ndarray] = {}
            for side in sides:
                wrist_pos = base_wrist[side] + _position_delta(args.position_mode, phase, args.position_amplitude)
                wrist_rot = base_rot[side] @ _wrist_delta(args.mode, phase, amp_rad)
                solved = retargeter.solve_wrist_pose(side, wrist_pos, wrist_rot)
                for n in _ARM_NAMES:
                    joints[n] = solved[n]
                _fill_target_values(joints, side, wrist_pos, wrist_rot)
                targets_for_graph[side] = wrist_pos
                metrics[side] = {
                    "pos_err_m": retargeter.last_error_m.get(f"{side}_wrist"),
                    "orient_err_deg": retargeter.last_wrist_orientation_error_deg.get(side),
                    "orient_debug": dict(retargeter.last_wrist_orientation_debug.get(side, {})),
                }

            # Fill inactive target markers from the reference pose so the
            # receiver's named-value merge never leaves stale VR targets.
            for side in ("left", "right"):
                if side not in sides:
                    _fill_target_values(joints, side, base_wrist[side], base_rot[side])

            sender.send([joints.get(n, 0.0) for n in names], time.time(), {side: True for side in sides})

            if now - last_log >= args.log_interval:
                last_log = now
                chunks = []
                for side in sides:
                    side_metrics = metrics.get(side, {})
                    pos_err_m = side_metrics.get("pos_err_m")
                    orient_err = side_metrics.get("orient_err_deg", float("nan"))
                    pos_err = float("nan") if pos_err_m is None else float(pos_err_m) * 1000.0
                    dbg = side_metrics.get("orient_debug", {})
                    clip = dbg.get("limit_clip")
                    q = dbg.get("clipped")
                    clip_s = ",".join(f"{float(v):.2f}" for v in clip) if clip is not None else "n/a"
                    q_s = ",".join(f"{float(v):.2f}" for v in q) if q is not None else "n/a"
                    chunks.append(
                        f"{side}: pos_err={pos_err:.1f}mm orient_err={orient_err:.1f}deg "
                        f"wrist_q=[{q_s}] clip=[{clip_s}]"
                    )
                print("[wrist_pose_test_sender] " + " | ".join(chunks))

            if graph is not None and now - last_graph >= graph_tick:
                last_graph = now
                positions, rotations = _graph_state(retargeter, sides, targets_for_graph)
                graph.update(positions, rotations)
    except KeyboardInterrupt:
        pass
    finally:
        sender.close()
        if graph is not None:
            graph.close()


if __name__ == "__main__":
    main()
