"""Replay a scripted DG5F thumb+index pinch motion into MuJoCo.

This bypasses Quest/AnyDex so the pinch pose itself can be tuned first.

Example:
    python -m scripts.pinch_motion_sender --send udp://127.0.0.1:6100 --hand both
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys
import time
import xml.etree.ElementTree as ET
from typing import Dict, Sequence, Tuple

import numpy as np

_ROOT = pathlib.Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.skeleton_graph import SkeletonGraphWindow  # noqa: E402
from teleop.articulation_protocol import ArticulationSender  # noqa: E402

_URDF = _ROOT / "rby1_dg5f.urdf"
_SIDES = ("left", "right")
_FINGERS = (1, 2, 3, 4, 5)
_SEGS = (1, 2, 3, 4)
_PINCH_CLOSED = {
    "left": {
        (1, 1): -0.890, (1, 2): 0.853, (1, 3): -0.500, (1, 4): 0.088,
        (2, 1): 0.371, (2, 2): 1.022, (2, 3): 0.821, (2, 4): 0.468,
    },
    "right": {
        (1, 1): -0.354, (1, 2): -2.214, (1, 3): 1.075, (1, 4): 1.571,
        (2, 1): -0.002, (2, 2): 1.096, (2, 3): 1.571, (2, 4): 1.571,
    },
}
_TIP_LINK = {
    "left": {1: "left_hand_ll_dg_1_tip", 2: "left_hand_ll_dg_2_tip"},
    "right": {1: "right_hand_rl_dg_1_tip", 2: "right_hand_rl_dg_2_tip"},
}


def _joint_name(side: str, finger: int, seg: int) -> str:
    hand = "lj" if side == "left" else "rj"
    return f"{side}_hand_{hand}_dg_{finger}_{seg}"


def _hand_joint_names(sides: Sequence[str]) -> list[str]:
    return [_joint_name(side, finger, seg) for side in sides for finger in _FINGERS for seg in _SEGS]


def _joint_limits(urdf_path: pathlib.Path, names: Sequence[str]) -> Dict[str, Tuple[float, float]]:
    wanted = set(names)
    limits: Dict[str, Tuple[float, float]] = {}
    for joint in ET.parse(urdf_path).getroot().iter("joint"):
        name = joint.get("name")
        if name not in wanted:
            continue
        limit = joint.find("limit")
        if limit is None:
            continue
        limits[name] = (float(limit.get("lower", "0")), float(limit.get("upper", "0")))
    missing = sorted(wanted - limits.keys())
    if missing:
        raise RuntimeError(f"{urdf_path}: missing joint limits for {missing[:8]}")
    return limits


def _clamp(value: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, value))


def _open_value(lo: float, hi: float) -> float:
    return _clamp(0.0, lo, hi)


def _closed_value(lo: float, hi: float) -> float:
    return hi if abs(hi) >= abs(lo) else lo


def _smoothstep(x: float) -> float:
    x = _clamp(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _norm3(v: np.ndarray) -> float:
    return math.sqrt(float(v[0]) * float(v[0]) + float(v[1]) * float(v[1]) + float(v[2]) * float(v[2]))


def _pinch_targets(
    names: Sequence[str],
    limits: Dict[str, Tuple[float, float]],
    amount: float,
    strength: float,
) -> Dict[str, float]:
    """Return full hand targets, with only thumb+index closing."""
    out: Dict[str, float] = {}
    for name in names:
        lo, hi = limits[name]
        open_q = _open_value(lo, hi)
        parts = name.rsplit("_", 2)
        side = "left" if name.startswith("left_") else "right"
        finger = int(parts[-2])
        seg = int(parts[-1])
        if (finger, seg) in _PINCH_CLOSED[side]:
            closed_q = _clamp(_PINCH_CLOSED[side][(finger, seg)], lo, hi)
            scale = strength * amount
            out[name] = _clamp(open_q + scale * (closed_q - open_q), lo, hi)
        else:
            out[name] = open_q
    return out


def _finger_joint_names(side: str, finger: int) -> list[str]:
    return [_joint_name(side, finger, seg) for seg in _SEGS]


def _fk_tip(tree, q: Dict[str, float], side: str, finger: int) -> np.ndarray:
    return tree.forward_kinematics(q, only=[_TIP_LINK[side][finger]])[_TIP_LINK[side][finger]].pos


def _solve_tip_ik(
    tree,
    q: Dict[str, float],
    side: str,
    finger: int,
    target: np.ndarray,
    limits: Dict[str, Tuple[float, float]],
    *,
    iterations: int = 50,
    damping: float = 0.01,
) -> None:
    """In-place DLS IK for one 4-DoF finger tip position target."""
    names = _finger_joint_names(side, finger)
    eps = 1e-4

    def solve3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Small explicit 3x3 solve to avoid native LAPACK crashes on bad frames."""
        a00, a01, a02 = float(a[0, 0]), float(a[0, 1]), float(a[0, 2])
        a10, a11, a12 = float(a[1, 0]), float(a[1, 1]), float(a[1, 2])
        a20, a21, a22 = float(a[2, 0]), float(a[2, 1]), float(a[2, 2])
        det = (
            a00 * (a11 * a22 - a12 * a21)
            - a01 * (a10 * a22 - a12 * a20)
            + a02 * (a10 * a21 - a11 * a20)
        )
        if abs(det) < 1e-12:
            return np.zeros(3)
        b0, b1, b2 = float(b[0]), float(b[1]), float(b[2])
        x0 = (
            b0 * (a11 * a22 - a12 * a21)
            - a01 * (b1 * a22 - a12 * b2)
            + a02 * (b1 * a21 - a11 * b2)
        ) / det
        x1 = (
            a00 * (b1 * a22 - a12 * b2)
            - b0 * (a10 * a22 - a12 * a20)
            + a02 * (a10 * b2 - b1 * a20)
        ) / det
        x2 = (
            a00 * (a11 * b2 - b1 * a21)
            - a01 * (a10 * b2 - b1 * a20)
            + b0 * (a10 * a21 - a11 * a20)
        ) / det
        return np.array([x0, x1, x2], dtype=float)

    for _ in range(iterations):
        p = _fk_tip(tree, q, side, finger)
        err = np.asarray(target) - p
        if _norm3(err) < 2e-4:
            break
        j = [[0.0 for _ in names] for _ in range(3)]
        for col, name in enumerate(names):
            old = q[name]
            q[name] = _clamp(old + eps, *limits[name])
            p2 = _fk_tip(tree, q, side, finger)
            q[name] = old
            diff = (p2 - p) / eps
            for row in range(3):
                j[row][col] = float(diff[row])
        # dq = J.T (J J.T + lambda^2 I)^-1 err
        a = np.zeros((3, 3), dtype=float)
        for r in range(3):
            for c in range(3):
                a[r, c] = sum(j[r][k] * j[c][k] for k in range(len(names)))
            a[r, r] += damping * damping
        y = solve3(a, err)
        dq = [
            sum(j[row][col] * float(y[row]) for row in range(3))
            for col in range(len(names))
        ]
        for name, delta in zip(names, dq):
            q[name] = _clamp(q[name] + float(delta), *limits[name])


def _ik_pinch_targets(
    side: str,
    amount: float,
    strength: float,
    tree,
    limits: Dict[str, Tuple[float, float]],
) -> Dict[str, float]:
    """Stage thumb/index tips toward a shared pinch point using FK IK."""
    prefix = "left_hand_lj_dg" if side == "left" else "right_hand_rj_dg"
    side_names = [f"{prefix}_{finger}_{seg}" for finger in _FINGERS for seg in _SEGS]
    q = {name: _open_value(*limits[name]) for name in side_names}

    # Use the old coordinate-descent solution only to define a good physical
    # contact point. Runtime targets are still generated by IK stage by stage.
    closed_seed = dict(q)
    for finger in (1, 2):
        for seg in _SEGS:
            name = _joint_name(side, finger, seg)
            closed_seed[name] = _clamp(_PINCH_CLOSED[side][(finger, seg)], *limits[name])
    pinch_point = 0.5 * (_fk_tip(tree, closed_seed, side, 1) + _fk_tip(tree, closed_seed, side, 2))

    open_thumb = _fk_tip(tree, q, side, 1)
    open_index = _fk_tip(tree, q, side, 2)
    a = _smoothstep(amount) * strength
    # Warm-start near the expected branch, but still solve tip targets by IK.
    for name in q:
        q[name] = q[name] + 0.8 * a * (closed_seed[name] - q[name])
    thumb_target = open_thumb + a * (pinch_point - open_thumb)
    index_target = open_index + a * (pinch_point - open_index)
    _solve_tip_ik(tree, q, side, 1, thumb_target, limits)
    _solve_tip_ik(tree, q, side, 2, index_target, limits)
    return {name: q[name] for name in side_names}


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


def _hand_skeleton(
    side: str,
    targets: Dict[str, float],
) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    sign = 1.0 if side == "left" else -1.0
    origin = np.array([0.0, sign * 0.16, 0.0])
    base_x = {1: -0.045, 2: -0.022, 3: 0.0, 4: 0.022, 5: 0.044}
    lengths = {
        1: (0.035, 0.027, 0.022, 0.018),
        2: (0.043, 0.030, 0.024, 0.018),
        3: (0.047, 0.034, 0.026, 0.019),
        4: (0.044, 0.031, 0.024, 0.018),
        5: (0.036, 0.026, 0.021, 0.016),
    }
    positions: Dict[str, np.ndarray] = {f"{side}_wrist": origin.copy()}
    rotations: Dict[str, np.ndarray] = {f"{side}_wrist": np.eye(3)}
    for finger in _FINGERS:
        p = origin + np.array([base_x[finger], sign * 0.025, 0.0])
        angle = 0.0
        prev = f"{side}_wrist"
        for seg, length in zip(_SEGS, lengths[finger]):
            q = targets[_joint_name(side, finger, seg)]
            angle += 0.55 * abs(q)
            direction = np.array([0.0, sign * math.cos(angle), -math.sin(angle)])
            p = p + length * direction
            name = f"{side}_f{finger}_{seg}"
            positions[name] = p.copy()
            rotations[name] = np.eye(3)
            prev = name
    return positions, rotations


def _graph_names_edges(sides: Sequence[str]) -> tuple[list[str], list[tuple[str, str]]]:
    names: list[str] = []
    edges: list[tuple[str, str]] = []
    for side in sides:
        wrist = f"{side}_wrist"
        names.append(wrist)
        for finger in _FINGERS:
            prev = wrist
            for seg in _SEGS:
                name = f"{side}_f{finger}_{seg}"
                names.append(name)
                edges.append((prev, name))
                prev = name
    return names, edges


def _parse_sides(value: str) -> tuple[str, ...]:
    if value == "both":
        return _SIDES
    if value in _SIDES:
        return (value,)
    raise argparse.ArgumentTypeError("expected left, right, or both")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--send", default="udp://127.0.0.1:6100",
                        help="mujoco_physics_process.py endpoint")
    parser.add_argument("--hand", type=_parse_sides, default="both",
                        help="left, right, or both")
    parser.add_argument("--hz", type=float, default=60.0)
    parser.add_argument("--strength", type=float, default=0.85,
                        help="0..1 fraction of thumb/index joint ranges")
    parser.add_argument("--close-time", type=float, default=0.22)
    parser.add_argument("--hold-time", type=float, default=0.8)
    parser.add_argument("--open-time", type=float, default=0.18)
    parser.add_argument("--open-hold-time", type=float, default=0.5)
    parser.add_argument("--static", action="store_true",
                        help="hold the fully closed IK pinch instead of cycling")
    parser.add_argument("--print-hz", type=float, default=2.0,
                        help="print FK thumb-tip/index-tip distance at this rate")
    parser.add_argument("--graph-hz", type=float, default=15.0)
    parser.add_argument("--no-graph", action="store_true")
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
        graph = SkeletonGraphWindow("Scripted DG5F thumb+index pinch", graph_names, graph_edges)

    print(
        f"[pinch_motion_sender] sending {len(names)} joints to {args.send} "
        f"hand={','.join(sides)} strength={args.strength:.2f}"
    )

    tick = 1.0 / args.hz
    graph_tick = 1.0 / args.graph_hz
    t0 = time.monotonic()
    last = 0.0
    last_graph = 0.0
    last_print = 0.0
    try:
        while True:
            now = time.monotonic()
            if now - last < tick:
                time.sleep(max(0.0, tick - (now - last)))
                continue
            last = now
            t = now - t0
            amount = 1.0 if args.static else _cycle_amount(
                t, args.close_time, args.hold_time, args.open_time, args.open_hold_time
            )
            targets: Dict[str, float] = {}
            for side in sides:
                targets.update(_ik_pinch_targets(side, amount, args.strength, hand_trees[side], limits))
            sender.send([targets[n] for n in names], time.time(), {side: True for side in sides})
            if now - last_print >= 1.0 / max(args.print_hz, 1e-6):
                last_print = now
                parts = []
                for side in sides:
                    dist = np.linalg.norm(
                        _fk_tip(hand_trees[side], targets, side, 1)
                        - _fk_tip(hand_trees[side], targets, side, 2)
                    )
                    parts.append(f"{side}={dist * 1000.0:.2f}mm")
                print(f"[pinch_motion_sender] amount={amount:.2f} tip_distance " + " ".join(parts))
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
