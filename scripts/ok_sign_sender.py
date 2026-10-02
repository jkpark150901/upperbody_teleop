"""Replay a scripted DG5F OK-sign motion into MuJoCo.

This bypasses Quest/AnyDex. Thumb and index are driven by the same staged
fingertip IK used by scripts.pinch_motion_sender; middle/ring/pinky stay
open so the resulting hand shape is an OK sign instead of a fist/pinch-only
test.

Example:
    python -m scripts.ok_sign_sender --send udp://127.0.0.1:6100 --hand both
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time
from typing import Dict, Sequence

import numpy as np

_ROOT = pathlib.Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.pinch_motion_sender import (  # noqa: E402
    _FINGERS,
    _SEGS,
    _SIDES,
    _URDF,
    _graph_names_edges,
    _hand_joint_names,
    _hand_skeleton,
    _ik_pinch_targets,
    _joint_limits,
    _joint_name,
    _open_value,
    _smoothstep,
)
from scripts.skeleton_graph import SkeletonGraphWindow  # noqa: E402
from teleop.articulation_protocol import ArticulationSender  # noqa: E402


def _parse_sides(value: str) -> tuple[str, ...]:
    if value == "both":
        return _SIDES
    if value in _SIDES:
        return (value,)
    raise argparse.ArgumentTypeError("expected left, right, or both")


def _cycle_amount(t: float, close_s: float, hold_s: float, open_s: float, open_hold_s: float) -> float:
    total = close_s + hold_s + open_s + open_hold_s
    u = t % total
    if u < close_s:
        return _smoothstep(u / close_s)
    u -= close_s
    if u < hold_s:
        return 1.0
    u -= hold_s
    if u < open_s:
        return 1.0 - _smoothstep(u / open_s)
    return 0.0


def _open_targets(
    sides: Sequence[str],
    limits: Dict[str, tuple[float, float]],
) -> Dict[str, float]:
    targets: Dict[str, float] = {}
    for side in sides:
        for finger in _FINGERS:
            for seg in _SEGS:
                name = _joint_name(side, finger, seg)
                targets[name] = _open_value(*limits[name])
    return targets


def _ok_targets(
    sides: Sequence[str],
    limits: Dict[str, tuple[float, float]],
    hand_trees: dict,
    *,
    amount: float,
    strength: float,
) -> Dict[str, float]:
    targets = _open_targets(sides, limits)
    for side in sides:
        ik_targets = _ik_pinch_targets(side, amount, strength, hand_trees[side], limits)
        prefix = "left_hand_lj_dg" if side == "left" else "right_hand_rj_dg"
        for finger in (1, 2):
            for seg in _SEGS:
                name = f"{prefix}_{finger}_{seg}"
                targets[name] = ik_targets[name]
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--send", default="udp://127.0.0.1:6100",
                        help="mujoco_physics_process.py endpoint")
    parser.add_argument("--hand", type=_parse_sides, default="both",
                        help="left, right, or both")
    parser.add_argument("--hz", type=float, default=60.0)
    parser.add_argument("--strength", type=float, default=1.0,
                        help="0..1 thumb/index OK-circle closure strength")
    parser.add_argument("--static", action="store_true",
                        help="hold the OK sign instead of looping open->OK->open")
    parser.add_argument("--close-time", type=float, default=0.35)
    parser.add_argument("--hold-time", type=float, default=1.2)
    parser.add_argument("--open-time", type=float, default=0.25)
    parser.add_argument("--open-hold-time", type=float, default=0.4)
    parser.add_argument("--graph-hz", type=float, default=15.0)
    parser.add_argument("--no-graph", action="store_true")
    parser.add_argument("--graph-labels", action="store_true",
                        help="show per-joint text labels in the skeleton graph")
    args = parser.parse_args()

    sides = args.hand
    names = _hand_joint_names(sides)
    limits = _joint_limits(_URDF, names)

    from robot_hand.urdf_fk import load_urdf
    from scripts.upper_body_retarget_vedo import _HAND_URDF_PATHS
    hand_trees = {side: load_urdf(str(_HAND_URDF_PATHS[side])) for side in sides}

    sender = ArticulationSender(args.send, names)
    graph = None
    if not args.no_graph:
        graph_names, graph_edges = _graph_names_edges(sides)
        graph = SkeletonGraphWindow(
            "Scripted DG5F OK-sign skeleton",
            graph_names,
            graph_edges,
            show_labels=args.graph_labels,
        )

    print(
        f"[ok_sign_sender] sending {len(names)} joints to {args.send} "
        f"hand={','.join(sides)} strength={args.strength:.2f} static={args.static}"
    )

    tick = 1.0 / args.hz
    graph_tick = 1.0 / args.graph_hz
    t0 = time.monotonic()
    last = 0.0
    last_graph = 0.0
    try:
        while True:
            now = time.monotonic()
            if now - last < tick:
                time.sleep(max(0.0, tick - (now - last)))
                continue
            last = now
            amount = 1.0 if args.static else _cycle_amount(
                now - t0, args.close_time, args.hold_time, args.open_time, args.open_hold_time
            )
            targets = _ok_targets(sides, limits, hand_trees, amount=amount, strength=args.strength)
            sender.send([targets[n] for n in names], time.time(), {side: True for side in sides})
            if graph is not None and now - last_graph >= graph_tick:
                last_graph = now
                positions: Dict[str, np.ndarray] = {}
                rotations: Dict[str, np.ndarray] = {}
                for side in sides:
                    p, r = _hand_skeleton(side, targets)
                    positions.update(p)
                    rotations.update(r)
                graph.update(positions, rotations)
    except KeyboardInterrupt:
        pass
    finally:
        sender.close()
        if graph is not None:
            graph.close()


if __name__ == "__main__":
    main()
