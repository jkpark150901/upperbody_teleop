"""Synthetic DG5F finger sender for MuJoCo gain tuning.

Sends direct finger joint targets over teleop/articulation_protocol.py to
scripts/mujoco_physics_process.py, while showing a small synthetic hand
skeleton in a matplotlib window. This bypasses Quest/SenseGlove retargeting
so actuator gain changes can be tested with a repeatable open/close motion.

Example:
    python -m scripts.finger_wave_sender --send udp://127.0.0.1:6100
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


def _joint_name(side: str, finger: int, seg: int) -> str:
    hand = "lj" if side == "left" else "rj"
    return f"{side}_hand_{hand}_dg_{finger}_{seg}"


def _finger_joint_names(sides: Sequence[str]) -> list[str]:
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
        raise RuntimeError(f"{urdf_path}: missing joint limits for {missing[:5]}...")
    return limits


def _clamp(value: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, value))


def _open_value(lo: float, hi: float) -> float:
    return _clamp(0.0, lo, hi)


def _closed_value(lo: float, hi: float) -> float:
    # Most DG5F flexion joints are one-sided. For ranges spanning zero,
    # the farther limit is the visible "curl" direction for this repeatable
    # tuning motion; this is deliberately generic across mirrored hands.
    return hi if abs(hi) >= abs(lo) else lo


def _finger_phase(finger: int, mode: str) -> float:
    if mode == "together":
        return 0.0
    if mode == "alternating":
        return 0.0 if finger in (1, 3, 5) else math.pi
    # wave: thumb starts first, pinky last.
    return (finger - 1) * 0.32


def _targets_at(
    t: float,
    names: Sequence[str],
    limits: Dict[str, Tuple[float, float]],
    *,
    period: float,
    amplitude: float,
    mode: str,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    omega = 2.0 * math.pi / period
    seg_weights = {1: 0.35, 2: 1.0, 3: 0.85, 4: 0.65}
    for name in names:
        parts = name.rsplit("_", 2)
        finger = int(parts[-2])
        seg = int(parts[-1])
        lo, hi = limits[name]
        open_q = _open_value(lo, hi)
        closed_q = _closed_value(lo, hi)
        # Smooth 0..1..0 profile. Keep endpoints soft to avoid injecting
        # step inputs when tuning tracking gains.
        flex = 0.5 - 0.5 * math.cos(omega * t - _finger_phase(finger, mode))
        scale = amplitude * seg_weights.get(seg, 1.0)
        out[name] = _clamp(open_q + scale * flex * (closed_q - open_q), lo, hi)
    return out


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
            rotations[name] = np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, math.cos(angle), -sign * math.sin(angle)],
                    [0.0, sign * math.sin(angle), math.cos(angle)],
                ],
                dtype=float,
            )
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
    parser.add_argument("--period", type=float, default=2.4,
                        help="seconds for one open-close cycle")
    parser.add_argument("--amplitude", type=float, default=0.65,
                        help="0=open only, 1=reaches the selected joint-limit endpoint")
    parser.add_argument("--mode", choices=("wave", "together", "alternating"), default="wave")
    parser.add_argument("--graph-hz", type=float, default=15.0,
                        help="skeleton graph refresh rate; UDP sending still uses --hz")
    parser.add_argument("--no-graph", action="store_true")
    args = parser.parse_args()

    sides = args.hand
    names = _finger_joint_names(sides)
    limits = _joint_limits(_URDF, names)
    sender = ArticulationSender(args.send, names)

    graph = None
    if not args.no_graph:
        graph_names, graph_edges = _graph_names_edges(sides)
        graph = SkeletonGraphWindow("Synthetic DG5F finger skeleton", graph_names, graph_edges)

    print(
        f"[finger_wave_sender] sending {len(names)} joints to {args.send} "
        f"hand={','.join(sides)} mode={args.mode} period={args.period}s amplitude={args.amplitude}"
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
            t = now - t0
            targets = _targets_at(
                t, names, limits,
                period=args.period, amplitude=args.amplitude, mode=args.mode,
            )
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
