"""Process 1 of 3: Quest hand-tracking retargeting.

Connects to the Quest 3 hand-tracking UDP stream, retargets each hand's 26
XR joints to DG5F finger joint angles, shows a live skeleton graph (x,y,z +
rx,ry,rz per XR joint, see scripts/skeleton_graph.py) and optionally
records the raw 26-joint stream to a .jsonl file -- but does NOT touch
MuJoCo at all. Retargeted joint angles go out over UDP (teleop/
articulation_protocol.py's ArticulationSender) to scripts/
mujoco_physics_process.py, which is the only thing that steps physics.

--arm-ik quest additionally solves each arm's shoulder/elbow/wrist from
the SAME Quest wrist pose (position + orientation), via robot_hand/
upper_body_retarget.py's solve_wrist_pose -- a VR-only alternative to
scripts/mocap_process.py's MocapApi-driven arms, for when you only have
the headset and no mocap suit. Don't run mocap_process.py at the same
time in that mode; both would send conflicting arm commands to the same
port.

Run all three processes with run_all.bat, or standalone:
    python -m scripts.hand_process --quest-host 192.168.8.156 --quest-port 5005 --send udp://127.0.0.1:6100
    python -m scripts.hand_process --arm-ik quest --quest-host 192.168.8.156 --quest-port 5005 --send udp://127.0.0.1:6100
"""
from __future__ import annotations

import argparse
import json
import pathlib
import queue
import sys
import threading
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
from robot_hand.sim_config import load_sim_config  # noqa: E402
from robot_hand.upper_body_retarget import RBY1UpperBodyRetargeter  # noqa: E402
from robot_hand.urdf_fk import load_urdf  # noqa: E402
from scripts.skeleton_graph import SkeletonGraphWindow  # noqa: E402
from scripts.pinch_motion_sender import (  # noqa: E402
    _URDF as _PINCH_URDF,
    _hand_joint_names as _pinch_hand_joint_names,
    _ik_pinch_targets,
    _joint_limits as _pinch_joint_limits,
)
from scripts.upper_body_retarget_mujoco import (  # noqa: E402
    QuestForwardCalibration, _quest_forward_reference, _quest_reach_limits,
)
from scripts.upper_body_retarget_vedo import (  # noqa: E402
    _HAND_JOINT_PREFIX, _HAND_URDF_PATHS, _URDF, _is_quest_hand,
)
from teleop.articulation_protocol import ArticulationSender  # noqa: E402

_ARM_NAMES = tuple(
    f"{side}_arm_{i}" for side in ("left", "right") for i in range(7)
) + ("torso_hp", "torso_5")

# Sent alongside _ARM_NAMES's joint angles as extra named values on the
# same UDP channel (teleop/articulation_protocol.py's sender/receiver
# already merge by name generically, no protocol change needed) -- the
# wrist IK's TARGET position, in the same robot-root/head-fixed world
# frame everything else renders in, not just the post-IK joint result.
# scripts/mujoco_physics_process.py draws a marker at this position so a
# target-vs-achieved gap (IK not fully converging, see solve_wrist's
# known local-minimum issue) is visible directly in the viewer instead of
# only inferable from numbers.
_WRIST_TARGET_NAMES = tuple(
    f"{side}_wrist_target_{axis}" for side in ("left", "right") for axis in ("x", "y", "z")
)

# The target ORIENTATION (full 3x3 rotation matrix, row-major, 9 floats)
# alongside the position above -- a plain sphere at the target position
# can't show rotation at all (no visible features), so
# mujoco_physics_process.py uses this to also draw a small oriented arrow,
# otherwise the marker looks static even while wrist orientation tracking
# is actually working.
_WRIST_TARGET_ROT_NAMES = tuple(
    f"{side}_wrist_target_r{row}{col}" for side in ("left", "right") for row in range(3) for col in range(3)
)


def _start_calibration_console(commands: "queue.SimpleQueue[float]") -> threading.Thread:
    """Same console-trigger pattern as scripts/upper_body_retarget_mujoco.py's
    _start_quest_calibration_console -- not imported from there since that
    one also starts printing calibration prompts tied to its own module
    state; this is a few lines, simpler to keep local."""
    def run() -> None:
        print("[hand_process] --arm-ik quest: type 'c' or 'calibrate' then Enter to capture "
              "forward-pose calibration after 3 seconds (arms forward 90deg, backs of hands up)")
        for line in sys.stdin:
            cmd = line.strip().lower()
            if cmd in {"c", "calibrate", "capture", "r", "recalibrate"}:
                commands.put(3.0)
            elif cmd:
                print("[hand_process] unknown command; use 'c' or 'calibrate'")

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread

_JOINT_NAMES = tuple(f"xr_{i}" for i in range(N_XR_JOINTS))
_EDGES = tuple(
    (f"xr_{a}", f"xr_{b}")
    for chain in _CHAIN.values()
    for a, b in zip((XR_WRIST,) + chain[:-1], chain)
)

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

    def write_frame(
        self,
        xr_joints: Dict[str, Optional[np.ndarray]],
        retargeted: Dict[str, float],
        hands_tracked: Optional[Dict[str, bool]] = None,
    ) -> None:
        now = time.monotonic()
        if self.t0 is None:
            self.t0 = now
        record = {
            "type": "frame",
            "t": now - self.t0,
            "xr_joints": {
                side: (None if j is None else j.tolist()) for side, j in xr_joints.items()
            },
            "hands_tracked": {side: bool((hands_tracked or {}).get(side)) for side in ("left", "right")},
            "retargeted": retargeted,
        }
        json.dump(record, self.file)
        self.file.write("\n")

    def close(self) -> None:
        self.file.close()


def _adaptive_filter_joints(
    raw: Dict[str, float],
    state: Dict[str, float],
    *,
    deadband: float,
    min_alpha: float,
    max_alpha: float,
    fast_delta: float,
) -> Dict[str, float]:
    """Suppress tiny retarget jitter while keeping large finger moves fast."""
    out: Dict[str, float] = {}
    span = max(1e-9, fast_delta - deadband)
    for name, value in raw.items():
        v = float(value)
        prev = state.get(name)
        if prev is None:
            filtered = v
        else:
            delta = abs(v - prev)
            if delta < deadband:
                filtered = prev
            else:
                blend = min(1.0, max(0.0, (delta - deadband) / span))
                alpha = min_alpha + (max_alpha - min_alpha) * blend
                filtered = prev + alpha * (v - prev)
        state[name] = filtered
        out[name] = filtered
    return out


def _closed_value(lo: float, hi: float) -> float:
    return hi if abs(hi) >= abs(lo) else lo


def _pinch_preset_targets(side: str, tree, strength: float) -> Dict[str, float]:
    """Thumb+index DG5F joint targets for a scripted pinch grip."""
    prefix = _HAND_JOINT_PREFIX[side]
    out: Dict[str, float] = {}
    for finger in (1, 2):
        for seg in (1, 2, 3, 4):
            name = f"{prefix}_{finger}_{seg}"
            joint = tree.joints[name]
            lo, hi = float(joint.lower), float(joint.upper)
            open_q = min(hi, max(lo, 0.0))
            closed_q = max(lo, min(hi, _PINCH_CLOSED[side][(finger, seg)]))
            out[name] = open_q + strength * (closed_q - open_q)
    return out


def _pinch_ik_targets(side: str, tree, limits: Dict[str, tuple[float, float]], strength: float) -> Dict[str, float]:
    """Thumb+index targets generated by staged fingertip IK."""
    raw = _ik_pinch_targets(side, 1.0, strength, tree, limits)
    prefix = _HAND_JOINT_PREFIX[side]
    keep = tuple(f"{prefix}_{finger}_" for finger in (1, 2))
    return {name: value for name, value in raw.items() if name.startswith(keep)}


def _apply_pinch_preset(
    retargeted: Dict[str, float],
    side: str,
    sample,
    preset: Dict[str, float],
    state: Dict[str, Dict[str, float | bool]],
    *,
    now: float,
    enter_distance: float,
    exit_distance: float,
    blend_time: float,
) -> Dict[str, float]:
    xr = sample.raw["xr_joints"]
    dist = float(np.linalg.norm(xr[5, :3] - xr[10, :3]))  # thumb tip <-> index tip
    s = state.setdefault(side, {"active": False, "blend": 0.0, "last_t": now})
    active = bool(s["active"])
    if active:
        active = dist < exit_distance
    else:
        active = dist < enter_distance
    dt = max(0.0, now - float(s["last_t"]))
    s["last_t"] = now
    s["active"] = active
    step = 1.0 if blend_time <= 1e-6 else dt / blend_time
    blend = float(s["blend"]) + (step if active else -step)
    blend = min(1.0, max(0.0, blend))
    s["blend"] = blend
    if blend <= 0.0:
        return retargeted
    smooth = blend * blend * (3.0 - 2.0 * blend)
    out = dict(retargeted)
    for name, target in preset.items():
        if name in out:
            out[name] = float(out[name] + smooth * (target - out[name]))
        else:
            out[name] = float(target)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quest-host", default="192.168.8.156")
    parser.add_argument("--quest-port", type=int, default=5005)
    parser.add_argument("--send", default="udp://127.0.0.1:6100",
                         help="scripts/mujoco_physics_process.py's --port, as udp://host:port")
    parser.add_argument("--sim-config", type=pathlib.Path, default=None,
                        help="path to sim_params.yaml; hand_retargeting in this file supplies defaults")
    parser.add_argument("--hand-retargeter", choices=("analytic", "anydex"), default=None)
    parser.add_argument("--anydex-mode", choices=("adaptive", "vector"), default=None,
                        help="AnyDex config profile from hand_retargeting.anydex_configs")
    parser.add_argument("--anydex-config-left", type=pathlib.Path, default=None)
    parser.add_argument("--anydex-config-right", type=pathlib.Path, default=None)
    parser.add_argument("--record", type=pathlib.Path, default=None, help="record raw xr_joints to this .jsonl file")
    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument(
        "--diagnose-retarget-latency", action="store_true",
        help="print Quest poll + hand retarget timing once a second. For --hand-retargeter anydex, "
        "this shows how long AnyDexRetarget takes to turn one OpenXR hand sample into DG5F joint angles.",
    )
    parser.add_argument(
        "--retarget-latency-log-interval", type=float, default=1.0,
        help="seconds between --diagnose-retarget-latency summaries",
    )
    parser.add_argument(
        "--finger-deadband", type=float, default=0.0,
        help="radians: hold finger targets when retargeted changes are smaller than this. "
        "0 disables the adaptive filter (default).",
    )
    parser.add_argument(
        "--finger-filter-min-alpha", type=float, default=0.45,
        help="adaptive finger target filter alpha for small non-deadband changes. "
        "Higher = faster but less smoothing.",
    )
    parser.add_argument(
        "--finger-filter-fast-delta", type=float, default=0.12,
        help="radians: finger target change size that bypasses most smoothing. "
        "Large quick gestures stay responsive.",
    )
    parser.add_argument(
        "--pinch-preset-grip", action="store_true", default=None,
        help="when Quest thumb/index tips form a pinch, override only DG5F thumb+index with a scripted pinch grip target.",
    )
    parser.add_argument(
        "--no-pinch-preset-grip", action="store_false", dest="pinch_preset_grip",
        help="disable the YAML-configured pinch override.",
    )
    parser.add_argument(
        "--pinch-grip-mode", choices=("preset", "ik"), default=None,
        help="with --pinch-preset-grip: preset uses stored joint angles; ik uses staged fingertip IK "
        "to generate the thumb/index grip target.",
    )
    parser.add_argument("--pinch-enter-distance", type=float, default=None,
                        help="metres: thumb/index tip distance below which preset pinch engages")
    parser.add_argument("--pinch-exit-distance", type=float, default=None,
                        help="metres: thumb/index tip distance above which preset pinch releases")
    parser.add_argument("--pinch-blend-time", type=float, default=None,
                        help="seconds to blend into/out of the preset pinch")
    parser.add_argument("--pinch-strength", type=float, default=None,
                        help="0..1 fraction of each thumb/index joint range used by the preset")
    parser.add_argument(
        "--arm-ik", choices=("none", "quest"), default="none",
        help="quest: also solve torso-fixed wrist-position+orientation IK per arm from the Quest "
        "wrist pose (robot_hand/upper_body_retarget.py's solve_wrist_pose), sent alongside finger "
        "joints -- see module docstring.",
    )
    parser.add_argument("--calibration-pose", choices=("attention", "tpose", "raise"), default="attention",
                         help="--arm-ik quest: RBY1UpperBodyRetargeter's calibration_pose")
    parser.add_argument("--wrist-scale", type=float, default=1.0,
                         help="--arm-ik quest: Quest-wrist-delta -> robot-wrist-delta scale multiplier")
    parser.add_argument(
        "--quest-max-reach", type=float, default=None,
        help="--arm-ik quest: max shoulder-to-wrist target radius in metres. Default is 98%% of the "
        "URDF upper+lower arm length. Targets beyond this are clamped onto that radius, same "
        "direction (ported from refer/HapticSyncController.cs's MaxRadius clamp).",
    )
    parser.add_argument(
        "--wrist-smoothing", type=float, default=1.0,
        help="--arm-ik quest: low-pass filter factor on the live wrist target (position delta + "
        "fingers/dorsal direction), ported from refer/HapticSyncController.cs's smoothingFactor. "
        "1.0 = no smoothing (default, matches prior behavior); lower = smoother/slower to follow.",
    )
    args = parser.parse_args()

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

    analytic_retargeters = {
        side: Dg5fHandRetargeter(load_urdf(str(_HAND_URDF_PATHS[side])), prefix)
        for side, prefix in _HAND_JOINT_PREFIX.items()
    }
    hand_trees = {side: load_urdf(str(_HAND_URDF_PATHS[side])) for side in ("left", "right")}
    pinch_strength = max(0.0, min(1.0, args.pinch_strength))
    if args.pinch_grip_mode == "ik":
        pinch_limits = _pinch_joint_limits(_PINCH_URDF, _pinch_hand_joint_names(("left", "right")))
        pinch_presets = {
            side: _pinch_ik_targets(side, hand_trees[side], pinch_limits, pinch_strength)
            for side in ("left", "right")
        }
    else:
        pinch_presets = {
            side: _pinch_preset_targets(side, hand_trees[side], pinch_strength)
            for side in ("left", "right")
        }
    anydex_retargeters = {}
    if args.hand_retargeter == "anydex":
        config_paths = {"left": args.anydex_config_left, "right": args.anydex_config_right}
        anydex_retargeters = {side: AnyDexDg5fRetargeter(side, config_paths[side]) for side in ("left", "right")}

    all_names = [name for r in analytic_retargeters.values() for names in r.chains.values() for name in names]

    # --arm-ik quest: same forward-reach calibration + torso-fixed wrist-
    # pose IK as scripts/upper_body_retarget_mujoco.py's --backend quest,
    # just sent out over UDP here instead of driving a local mj_forward
    # puppet -- see that module for the IK/calibration details
    # (QuestForwardCalibration, solve_wrist_pose). last_arm_joints holds
    # the last successful solve per side so a momentarily-untracked or
    # out-of-reach hand keeps sending its last pose instead of snapping to
    # 0 (all_names needs a value every tick once --arm-ik quest is on).
    arm_retargeter = None
    quest_calibration = None
    quest_calibration_commands: Optional["queue.SimpleQueue[float]"] = None
    quest_shoulder: Dict[str, np.ndarray] = {}
    quest_reach_limit: Dict[str, float] = {}
    last_arm_joints: Dict[str, float] = {}
    last_wrist_targets: Dict[str, float] = {}
    filtered_hand_joints: Dict[str, float] = {}
    pinch_state: Dict[str, Dict[str, float | bool]] = {}
    last_reach_warning: Dict[str, float] = {"left": 0.0, "right": 0.0}
    last_orient_log: Dict[str, float] = {"left": 0.0, "right": 0.0}
    last_status_log = [0.0]
    was_calibrated = False
    if args.arm_ik == "quest":
        arm_tree = load_urdf(str(_URDF))
        arm_retargeter = RBY1UpperBodyRetargeter(arm_tree, calibration_pose=args.calibration_pose)
        base_wrist, base_rot = _quest_forward_reference(arm_retargeter)
        quest_shoulder, quest_reach_limit = _quest_reach_limits(arm_retargeter, args.quest_max_reach)
        head_pose_ref = arm_retargeter.poses()["link_head_2"]
        head_pose = (head_pose_ref.pos.copy(), head_pose_ref.as_matrix())
        quest_calibration = QuestForwardCalibration(
            base_wrist, base_rot, quest_reach_limit, head_pose, args.wrist_smoothing,
        )
        quest_calibration_commands = queue.SimpleQueue()
        _start_calibration_console(quest_calibration_commands)
        last_arm_joints = {n: arm_retargeter._values(arm_retargeter.q)[n] for n in _ARM_NAMES}
        forward_reach_joints = dict(last_arm_joints)  # the reference pose itself, reapplied on every (re)calibration
        last_wrist_targets = {
            f"{side}_wrist_target_{axis}": float(base_wrist[side][i])
            for side in ("left", "right") for i, axis in enumerate(("x", "y", "z"))
        }
        for side in ("left", "right"):
            for row in range(3):
                for col in range(3):
                    last_wrist_targets[f"{side}_wrist_target_r{row}{col}"] = float(base_rot[side][row, col])
        forward_reach_wrist_targets = dict(last_wrist_targets)
        all_names = all_names + list(_ARM_NAMES) + list(_WRIST_TARGET_NAMES) + list(_WRIST_TARGET_ROT_NAMES)
        print(
            "[hand_process] --arm-ik quest reach limits: "
            + ", ".join(f"{side}={quest_reach_limit[side]:.3f}m" for side in ("left", "right"))
        )

    reader = QuestHandUDPReader(args.quest_host, args.quest_port)
    reader.connect()
    sender = ArticulationSender(args.send, all_names)
    recorder = HandRecorder(args.record) if args.record else None
    graph = SkeletonGraphWindow("Quest hand skeleton (x,y,z / rx,ry,rz)", _JOINT_NAMES, _EDGES)

    print(f"[hand_process] Quest UDP {args.quest_host}:{args.quest_port} -> {args.send}"
          + (f", recording to {args.record}" if args.record else ""))
    print(
        "[hand_process] "
        f"retargeter={args.hand_retargeter}"
        + (
            f" anydex_mode={args.anydex_mode} "
            f"left_config={args.anydex_config_left} right_config={args.anydex_config_right}"
            if args.hand_retargeter == "anydex" else ""
        )
        + " | "
        f"pinch_override enabled={bool(args.pinch_preset_grip)} mode={args.pinch_grip_mode} "
        f"enter={args.pinch_enter_distance:.3f}m exit={args.pinch_exit_distance:.3f}m "
        f"blend={args.pinch_blend_time:.3f}s strength={args.pinch_strength:.2f}"
    )

    tick_interval = 1.0 / args.hz
    last_tick = 0.0
    last_hand: Dict[str, object] = {"left": None, "right": None}
    last_hand_joints: Dict[str, float] = {}
    latency_stats = {
        "poll_count": 0,
        "poll_sum_ms": 0.0,
        "poll_max_ms": 0.0,
        "last_log": time.monotonic(),
        "side": {
            side: {
                "count": 0,
                "fresh": 0,
                "retarget_sum_ms": 0.0,
                "retarget_max_ms": 0.0,
                "age_done_sum_ms": 0.0,
                "age_done_max_ms": 0.0,
                "age_send_sum_ms": 0.0,
                "age_send_max_ms": 0.0,
                "last_seq": None,
                "last_retarget_ms": None,
                "last_age_done_ms": None,
                "last_age_send_ms": None,
            }
            for side in ("left", "right")
        },
    }

    def record_poll_latency(ms: float) -> None:
        if not args.diagnose_retarget_latency:
            return
        latency_stats["poll_count"] += 1
        latency_stats["poll_sum_ms"] += ms
        latency_stats["poll_max_ms"] = max(latency_stats["poll_max_ms"], ms)

    def record_retarget_latency(side: str, sample, retarget_ms: float, done_wall_time: float) -> None:
        if not args.diagnose_retarget_latency:
            return
        s = latency_stats["side"][side]
        seq = sample.raw.get("seq") if getattr(sample, "raw", None) else None
        s["fresh"] += 1
        s["last_seq"] = seq
        t_recv_host = sample.raw.get("t_recv_host", sample.timestamp)
        age_done_ms = (done_wall_time - t_recv_host) * 1000.0
        s["count"] += 1
        s["retarget_sum_ms"] += retarget_ms
        s["retarget_max_ms"] = max(s["retarget_max_ms"], retarget_ms)
        s["age_done_sum_ms"] += age_done_ms
        s["age_done_max_ms"] = max(s["age_done_max_ms"], age_done_ms)
        s["last_retarget_ms"] = retarget_ms
        s["last_age_done_ms"] = age_done_ms

    def record_send_latency(side: str, sample, send_wall_time: float) -> None:
        if not args.diagnose_retarget_latency:
            return
        s = latency_stats["side"][side]
        t_recv_host = sample.raw.get("t_recv_host", sample.timestamp)
        age_send_ms = (send_wall_time - t_recv_host) * 1000.0
        s["age_send_sum_ms"] += age_send_ms
        s["age_send_max_ms"] = max(s["age_send_max_ms"], age_send_ms)
        s["last_age_send_ms"] = age_send_ms

    def maybe_log_retarget_latency(now: float) -> None:
        if not args.diagnose_retarget_latency:
            return
        if now - latency_stats["last_log"] < args.retarget_latency_log_interval:
            return
        elapsed = max(1e-9, now - latency_stats["last_log"])
        latency_stats["last_log"] = now
        poll_count = latency_stats["poll_count"]
        poll_avg = latency_stats["poll_sum_ms"] / poll_count if poll_count else 0.0
        parts = [
            f"[retarget-latency] backend={args.hand_retargeter}",
            f"poll avg/max={poll_avg:.2f}/{latency_stats['poll_max_ms']:.2f}ms n={poll_count}",
        ]
        for side in ("left", "right"):
            s = latency_stats["side"][side]
            count = s["count"]
            if count:
                rt_avg = s["retarget_sum_ms"] / count
                done_avg = s["age_done_sum_ms"] / count
                send_avg = s["age_send_sum_ms"] / count if s["age_send_sum_ms"] else 0.0
                parts.append(
                    f"{side}: rt avg/max/last={rt_avg:.2f}/{s['retarget_max_ms']:.2f}/"
                    f"{s['last_retarget_ms']:.2f}ms "
                    f"age_done avg/max/last={done_avg:.1f}/{s['age_done_max_ms']:.1f}/"
                    f"{s['last_age_done_ms']:.1f}ms "
                    f"age_send avg/max/last={send_avg:.1f}/{s['age_send_max_ms']:.1f}/"
                    f"{s['last_age_send_ms']:.1f}ms "
                    f"fresh_retargets={count} fresh_seq={s['fresh']} rate={count / elapsed:.1f}Hz"
                )
            else:
                parts.append(f"{side}: no tracked samples")
            for key in (
                "count", "fresh", "retarget_sum_ms", "retarget_max_ms",
                "age_done_sum_ms", "age_done_max_ms", "age_send_sum_ms", "age_send_max_ms",
            ):
                s[key] = 0 if key in ("count", "fresh") else 0.0
        latency_stats["poll_count"] = 0
        latency_stats["poll_sum_ms"] = 0.0
        latency_stats["poll_max_ms"] = 0.0
        print(" | ".join(parts))

    try:
        while True:
            now = time.monotonic()
            if now - last_tick < tick_interval:
                time.sleep(max(0.0, tick_interval - (now - last_tick)))
                continue
            last_tick = now

            poll_t0 = time.perf_counter()
            left, right = reader.poll()
            record_poll_latency((time.perf_counter() - poll_t0) * 1000.0)
            fresh_hand: Dict[str, object] = {}
            if left is not None:
                last_hand["left"] = left
                fresh_hand["left"] = left
            if right is not None:
                last_hand["right"] = right
                fresh_hand["right"] = right

            if arm_retargeter is not None:
                while True:
                    try:
                        quest_calibration.request_capture(quest_calibration_commands.get_nowait())
                    except queue.Empty:
                        break
                calibrated = quest_calibration.update(last_hand, now)
                if calibrated and not was_calibrated:
                    # Just (re)calibrated -- reset the robot to the same
                    # arms-forward reference pose the calibration itself is
                    # anchored to, instead of silently keeping whatever
                    # pose the arm happened to be in (e.g. wherever live
                    # tracking last left it, on a recalibration). Without
                    # this the operator has no robot-side cue of where the
                    # reference actually is, and their own physical pose
                    # may no longer match the (possibly stale) one assumed
                    # at the 3-second capture.
                    last_arm_joints = dict(forward_reach_joints)
                    last_wrist_targets = dict(forward_reach_wrist_targets)
                    print("[hand_process] robot reset to arms-forward reference pose")
                was_calibrated = calibrated
                if now - last_status_log[0] > 2.0:
                    last_status_log[0] = now
                    print(
                        f"[hand_process] arm-ik status: calibrated={calibrated} "
                        + ", ".join(
                            f"{side}_is_quest_hand={_is_quest_hand(last_hand.get(side))}"
                            for side in ("left", "right")
                        )
                    )
                if calibrated:
                    for side in ("left", "right"):
                        sample = last_hand.get(side)
                        if not _is_quest_hand(sample):
                            continue
                        wrist_pos, wrist_rotation = quest_calibration.wrist_target(side, sample, args.wrist_scale)
                        for i, axis in enumerate(("x", "y", "z")):
                            last_wrist_targets[f"{side}_wrist_target_{axis}"] = float(wrist_pos[i])
                        for row in range(3):
                            for col in range(3):
                                last_wrist_targets[f"{side}_wrist_target_r{row}{col}"] = float(wrist_rotation[row, col])
                        # Direction-preserving clamp onto the max-reach sphere
                        # (ported from refer/HapticSyncController.cs:
                        # vectorFromRoot.normalized * MaxRadius) instead of
                        # dropping the frame and holding the previous pose --
                        # keeps following smoothly right up to the limit
                        # instead of snapping/freezing the instant it's
                        # exceeded.
                        from_shoulder = wrist_pos - quest_shoulder[side]
                        reach = float(np.linalg.norm(from_shoulder))
                        if reach > quest_reach_limit[side]:
                            if now - last_reach_warning[side] > 1.0:
                                last_reach_warning[side] = now
                                print(
                                    f"[hand_process] {side} wrist target clamped to max reach "
                                    f"({reach:.3f}m -> {quest_reach_limit[side]:.3f}m)"
                                )
                            wrist_pos = quest_shoulder[side] + from_shoulder * (quest_reach_limit[side] / reach)
                        arm_joints_full = arm_retargeter.solve_wrist_pose(side, wrist_pos, wrist_rotation)
                        for n in _ARM_NAMES:
                            last_arm_joints[n] = arm_joints_full[n]
                        if now - last_orient_log[side] > 2.0:
                            last_orient_log[side] = now
                            pos_err = arm_retargeter.last_error_m.get(f"{side}_wrist")
                            orient_err = arm_retargeter.last_wrist_orientation_error_deg.get(side)
                            orient_dbg = arm_retargeter.last_wrist_orientation_debug.get(side, {})
                            if pos_err is not None and orient_err is not None:
                                clip = orient_dbg.get("limit_clip")
                                clipped = orient_dbg.get("clipped")
                                clip_s = (
                                    " clip(rad)=" + ",".join(f"{float(v):.2f}" for v in clip)
                                    if clip is not None else ""
                                )
                                joints_s = (
                                    " wrist_q(rad)=" + ",".join(f"{float(v):.2f}" for v in clipped)
                                    if clipped is not None else ""
                                )
                                print(
                                    f"[hand_process] {side} wrist IK: pos_err={pos_err*1000:.1f}mm "
                                    f"orient_err={orient_err:.1f}deg{joints_s}{clip_s}"
                                )
                            else:
                                print(f"[hand_process] {side} wrist IK: no error diagnostic yet")

            joints: Dict[str, float] = dict(last_arm_joints)
            joints.update(last_wrist_targets)
            joints.update(last_hand_joints)
            xr_joints_by_side: Dict[str, Optional[np.ndarray]] = {"left": None, "right": None}
            positions: Dict[str, np.ndarray] = {}
            rotations: Dict[str, np.ndarray] = {}
            for side, sample in fresh_hand.items():
                if sample is None:
                    continue
                xr = sample.raw["xr_joints"]
                xr_joints_by_side[side] = xr
                retarget_t0 = time.perf_counter()
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
                    retargeted = _apply_pinch_preset(
                        retargeted,
                        side,
                        sample,
                        pinch_presets[side],
                        pinch_state,
                        now=now,
                        enter_distance=args.pinch_enter_distance,
                        exit_distance=args.pinch_exit_distance,
                        blend_time=args.pinch_blend_time,
                    )
                retarget_done_wall = time.time()
                record_retarget_latency(
                    side, sample, (time.perf_counter() - retarget_t0) * 1000.0, retarget_done_wall,
                )
                last_hand_joints.update(retargeted)
                joints.update(retargeted)
            for side, sample in last_hand.items():
                if sample is None:
                    continue
                xr = sample.raw["xr_joints"]
                xr_joints_by_side[side] = xr
                for i in range(N_XR_JOINTS):
                    positions[f"xr_{i}"] = xr[i, :3]
                    rotations[f"xr_{i}"] = _quat_xyzw_to_matrix(xr[i, 3:])

            if joints:
                send_wall_time = time.time()
                sender.send([joints.get(n, 0.0) for n in all_names], send_wall_time, {
                    "left": last_hand["left"] is not None, "right": last_hand["right"] is not None,
                })
                for side, sample in fresh_hand.items():
                    if sample is not None:
                        record_send_latency(side, sample, send_wall_time)
            if recorder is not None:
                recorder.write_frame(
                    xr_joints_by_side,
                    joints,
                    {side: side in fresh_hand for side in ("left", "right")},
                )
            if positions:
                graph.update(positions, rotations)
            maybe_log_retarget_latency(now)
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
