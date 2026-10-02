"""Replay a recorded Quest/OpenXR hand skeleton into MuJoCo.

Reads the JSONL made by:
    python -m scripts.hand_process --record recordings/quest_hand_latest.jsonl

Then runs the same hand retargeting backend configured in configs/sim_params.yaml
and sends DG5F finger joint targets to scripts.mujoco_physics_process over the
normal articulation UDP channel. This lets a captured Quest pinch/OK motion be
checked in simulation without wearing the headset again.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from types import SimpleNamespace
from typing import Dict, Iterable, Optional

import numpy as np

_ROOT = pathlib.Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from devices.quest_hand.quest_hand_reader import N_XR_JOINTS, XR_WRIST, _CHAIN, _quat_xyzw_to_matrix  # noqa: E402
from robot_hand.anydex_retarget import AnyDexDg5fRetargeter  # noqa: E402
from robot_hand.hand_retarget import Dg5fHandRetargeter  # noqa: E402
from robot_hand.sim_config import load_sim_config  # noqa: E402
from robot_hand.urdf_fk import load_urdf  # noqa: E402
from scripts.hand_process import (  # noqa: E402
    _HAND_JOINT_PREFIX,
    _HAND_URDF_PATHS,
    _PINCH_URDF,
    _adaptive_filter_joints,
    _apply_pinch_preset,
    _pinch_hand_joint_names,
    _pinch_ik_targets,
    _pinch_joint_limits,
    _pinch_preset_targets,
)
from scripts.skeleton_graph import SkeletonGraphWindow  # noqa: E402
from teleop.articulation_protocol import ArticulationSender  # noqa: E402

_SIDES = ("left", "right")
_JOINT_NAMES = tuple(f"{side}_xr_{i}" for side in _SIDES for i in range(N_XR_JOINTS))
_EDGES = tuple(
    (f"{side}_xr_{a}", f"{side}_xr_{b}")
    for side in _SIDES
    for chain in _CHAIN.values()
    for a, b in zip((XR_WRIST,) + chain[:-1], chain)
)
_SIDE_OFFSET = {
    "left": np.array([0.0, 0.14, 0.0]),
    "right": np.array([0.0, -0.14, 0.0]),
}
_THUMB_TIP = 5
_INDEX_TIP = 10


def _load_frames(path: pathlib.Path) -> list[dict]:
    frames = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("type") == "header" and record.get("format") != "hand_xr_v1":
                raise RuntimeError(f"{path}: expected hand_xr_v1 recording, got {record.get('format')!r}")
            if record.get("type") == "frame":
                frames.append(record)
    if not frames:
        raise RuntimeError(f"{path}: no frame records found")
    return frames


def _iter_frame_indices(num_frames: int, loop: bool) -> Iterable[int]:
    while True:
        for i in range(num_frames):
            yield i
        if not loop:
            return


def _resolve_hand_config(args: argparse.Namespace) -> dict:
    sim_config = load_sim_config(args.sim_config)
    hand_config = sim_config.get("hand_retargeting", {})
    pinch_config = hand_config.get("pinch_override", {})

    if args.hand_retargeter is None:
        args.hand_retargeter = str(hand_config.get("backend", "analytic"))
    if args.hand_retargeter not in ("analytic", "anydex"):
        raise ValueError(f"hand_retargeting.backend must be 'analytic' or 'anydex', got {args.hand_retargeter!r}")

    if args.anydex_mode is None:
        args.anydex_mode = str(hand_config.get("anydex_mode", "adaptive"))
    if args.anydex_mode not in ("adaptive", "vector"):
        raise ValueError(f"hand_retargeting.anydex_mode must be 'adaptive' or 'vector', got {args.anydex_mode!r}")

    anydex_configs = hand_config.get("anydex_configs", {})
    selected_anydex = anydex_configs.get(args.anydex_mode, {})
    if args.anydex_config_left is None:
        if selected_anydex.get("left"):
            args.anydex_config_left = pathlib.Path(selected_anydex["left"])
        elif hand_config.get("anydex_config_left"):
            args.anydex_config_left = pathlib.Path(hand_config["anydex_config_left"])
    if args.anydex_config_right is None:
        if selected_anydex.get("right"):
            args.anydex_config_right = pathlib.Path(selected_anydex["right"])
        elif hand_config.get("anydex_config_right"):
            args.anydex_config_right = pathlib.Path(hand_config["anydex_config_right"])

    if args.pinch_preset_grip is None:
        args.pinch_preset_grip = bool(pinch_config.get("enabled", False))
    if args.pinch_grip_mode is None:
        args.pinch_grip_mode = str(pinch_config.get("mode", "preset"))
    if args.pinch_grip_mode not in ("preset", "ik"):
        raise ValueError(f"hand_retargeting.pinch_override.mode must be 'preset' or 'ik', got {args.pinch_grip_mode!r}")
    if args.pinch_enter_distance is None:
        args.pinch_enter_distance = float(pinch_config.get("enter_distance", 0.035))
    if args.pinch_exit_distance is None:
        args.pinch_exit_distance = float(pinch_config.get("exit_distance", 0.055))
    if args.pinch_blend_time is None:
        args.pinch_blend_time = float(pinch_config.get("blend_time", 0.18))
    if args.pinch_strength is None:
        args.pinch_strength = float(pinch_config.get("strength", 0.85))

    return hand_config


def _graph_positions(xr_by_side: dict[str, Optional[np.ndarray]]) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    positions: Dict[str, np.ndarray] = {}
    rotations: Dict[str, np.ndarray] = {}
    for side, xr in xr_by_side.items():
        if xr is None:
            continue
        wrist = xr[XR_WRIST, :3]
        for i in range(N_XR_JOINTS):
            key = f"{side}_xr_{i}"
            positions[key] = xr[i, :3] - wrist + _SIDE_OFFSET[side]
            if xr.shape[1] >= 7:
                rotations[key] = _quat_xyzw_to_matrix(xr[i, 3:])
    return positions, rotations


def _pinch_distance_mm(xr: Optional[np.ndarray]) -> str:
    if xr is None:
        return "NA"
    dist = float(np.linalg.norm(xr[_THUMB_TIP, :3] - xr[_INDEX_TIP, :3]))
    return f"{dist * 1000.0:5.1f}mm"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=pathlib.Path, default=pathlib.Path("recordings/quest_hand_latest.jsonl"))
    parser.add_argument("--send", default="udp://127.0.0.1:6100")
    parser.add_argument("--sim-config", type=pathlib.Path, default=None)
    parser.add_argument("--hand-retargeter", choices=("analytic", "anydex"), default=None)
    parser.add_argument("--anydex-mode", choices=("adaptive", "vector"), default=None)
    parser.add_argument("--anydex-config-left", type=pathlib.Path, default=None)
    parser.add_argument("--anydex-config-right", type=pathlib.Path, default=None)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--start-time", type=float, default=0.0,
                        help="recording time in seconds to start replay from")
    parser.add_argument("--duration", type=float, default=None,
                        help="seconds to replay after --start-time; default replays to the end")
    parser.add_argument(
        "--source", choices=("retargeted", "retarget"), default="retargeted",
        help="retargeted: replay joint commands saved in the recording. "
        "retarget: recompute from raw xr_joints with the selected backend.",
    )
    parser.add_argument("--no-graph", action="store_true")
    parser.add_argument("--labels", action="store_true")
    parser.add_argument("--print-hz", type=float, default=10.0)
    parser.add_argument("--finger-deadband", type=float, default=0.0)
    parser.add_argument("--finger-filter-min-alpha", type=float, default=0.45)
    parser.add_argument("--finger-filter-fast-delta", type=float, default=0.12)
    parser.add_argument("--pinch-preset-grip", action="store_true", default=None)
    parser.add_argument("--no-pinch-preset-grip", action="store_false", dest="pinch_preset_grip")
    parser.add_argument("--pinch-grip-mode", choices=("preset", "ik"), default=None)
    parser.add_argument("--pinch-enter-distance", type=float, default=None)
    parser.add_argument("--pinch-exit-distance", type=float, default=None)
    parser.add_argument("--pinch-blend-time", type=float, default=None)
    parser.add_argument("--pinch-strength", type=float, default=None)
    args = parser.parse_args()

    _resolve_hand_config(args)
    frames = _load_frames(args.file)
    start_time = max(0.0, float(args.start_time))
    end_time = None if args.duration is None else start_time + max(0.0, float(args.duration))
    frames = [
        frame for frame in frames
        if float(frame.get("t", 0.0)) >= start_time
        and (end_time is None or float(frame.get("t", 0.0)) <= end_time)
    ]
    if not frames:
        raise RuntimeError(
            f"{args.file}: no frames in replay window start={start_time:.3f}s "
            + ("to end" if end_time is None else f"duration={args.duration:.3f}s")
        )
    t0_window = float(frames[0].get("t", 0.0))
    for frame in frames:
        frame["t"] = float(frame.get("t", 0.0)) - t0_window

    analytic_retargeters = {
        side: Dg5fHandRetargeter(load_urdf(str(_HAND_URDF_PATHS[side])), prefix)
        for side, prefix in _HAND_JOINT_PREFIX.items()
    }
    hand_trees = {side: load_urdf(str(_HAND_URDF_PATHS[side])) for side in _SIDES}
    all_names = [name for r in analytic_retargeters.values() for names in r.chains.values() for name in names]

    anydex_retargeters = {}
    if args.source == "retarget" and args.hand_retargeter == "anydex":
        config_paths = {"left": args.anydex_config_left, "right": args.anydex_config_right}
        anydex_retargeters = {side: AnyDexDg5fRetargeter(side, config_paths[side]) for side in _SIDES}

    pinch_presets: dict[str, dict[str, float]] = {}
    if args.source == "retarget" and args.pinch_preset_grip:
        pinch_strength = max(0.0, min(1.0, args.pinch_strength))
        if args.pinch_grip_mode == "ik":
            pinch_limits = _pinch_joint_limits(_PINCH_URDF, _pinch_hand_joint_names(_SIDES))
            pinch_presets = {
                side: _pinch_ik_targets(side, hand_trees[side], pinch_limits, pinch_strength)
                for side in _SIDES
            }
        else:
            pinch_presets = {
                side: _pinch_preset_targets(side, hand_trees[side], pinch_strength)
                for side in _SIDES
            }

    sender = ArticulationSender(args.send, all_names)
    graph = None if args.no_graph else SkeletonGraphWindow(
        f"Quest recording -> MuJoCo: {args.file.name}",
        _JOINT_NAMES,
        _EDGES,
        show_labels=args.labels,
    )
    filtered_hand_joints: Dict[str, float] = {}
    pinch_state: Dict[str, Dict[str, float | bool]] = {}
    last_hand_joints: Dict[str, float] = {}
    last_log = 0.0

    print(
        f"[replay_quest_hand_to_mujoco] loaded {len(frames)} frames from {args.file} "
        f"(window start={start_time:.3f}s"
        + (")" if end_time is None else f" duration={args.duration:.3f}s)")
    )
    print(
        "[replay_quest_hand_to_mujoco] "
        f"source={args.source} retargeter={args.hand_retargeter}"
        + (
            f" anydex_mode={args.anydex_mode} left_config={args.anydex_config_left} "
            f"right_config={args.anydex_config_right}"
            if args.hand_retargeter == "anydex" else ""
        )
        + " | "
        f"pinch_override enabled={bool(args.pinch_preset_grip)} mode={args.pinch_grip_mode}"
    )
    print(f"[replay_quest_hand_to_mujoco] sending hand joints to {args.send}")

    try:
        prev_t = float(frames[0].get("t", 0.0))
        for idx in _iter_frame_indices(len(frames), args.loop):
            frame = frames[idx]
            t = float(frame.get("t", prev_t))
            if idx > 0:
                time.sleep(max(0.0, (t - prev_t) / max(args.speed, 1e-6)))
            prev_t = t

            xr_by_side: dict[str, Optional[np.ndarray]] = {"left": None, "right": None}
            joints: Dict[str, float] = dict(last_hand_joints)
            hands = {"left": False, "right": False}
            if args.source == "retargeted":
                saved = frame.get("retargeted", {})
                if isinstance(saved, dict):
                    for name, value in saved.items():
                        if name in all_names:
                            joints[name] = float(value)
                            last_hand_joints[name] = float(value)
            for side in _SIDES:
                raw = frame.get("xr_joints", {}).get(side)
                if raw is None:
                    continue
                xr = np.asarray(raw, dtype=float)
                if xr.shape[0] != N_XR_JOINTS or xr.shape[1] < 3:
                    continue
                xr_by_side[side] = xr
                tracked = frame.get("hands_tracked", {}).get(side, True)
                hands[side] = bool(tracked)
                if args.source == "retargeted":
                    continue
                if not tracked:
                    continue
                if args.hand_retargeter == "anydex":
                    retargeted = anydex_retargeters[side].retarget_openxr_joints(xr)
                else:
                    retargeted = analytic_retargeters[side].retarget_openxr_joints(xr)
                if args.finger_deadband > 0.0:
                    retargeted = _adaptive_filter_joints(
                        retargeted,
                        filtered_hand_joints,
                        deadband=args.finger_deadband,
                        min_alpha=args.finger_filter_min_alpha,
                        max_alpha=1.0,
                        fast_delta=args.finger_filter_fast_delta,
                    )
                if args.pinch_preset_grip:
                    sample = SimpleNamespace(raw={"xr_joints": xr})
                    retargeted = _apply_pinch_preset(
                        retargeted,
                        side,
                        sample,
                        pinch_presets[side],
                        pinch_state,
                        now=time.monotonic(),
                        enter_distance=args.pinch_enter_distance,
                        exit_distance=args.pinch_exit_distance,
                        blend_time=args.pinch_blend_time,
                    )
                last_hand_joints.update(retargeted)
                joints.update(retargeted)

            sender.send([joints.get(name, 0.0) for name in all_names], time.time(), hands)
            if graph is not None:
                positions, rotations = _graph_positions(xr_by_side)
                if positions:
                    graph.update(positions, rotations)

            now = time.monotonic()
            if now - last_log >= 1.0 / max(args.print_hz, 1e-6):
                last_log = now
                print(
                    f"[quest-replay] frame={idx:05d} t={t:7.3f}s "
                    f"left={_pinch_distance_mm(xr_by_side['left'])} "
                    f"right={_pinch_distance_mm(xr_by_side['right'])}"
                )
    except KeyboardInterrupt:
        pass
    finally:
        sender.close()
        if graph is not None:
            graph.close()


if __name__ == "__main__":
    main()
