"""Quest 3 hand tracking, visualized before the rest of the pipeline exists:
the raw 26-joint OpenXR skeleton (what's actually arriving over the wire)
next to the DG5F URDF hand it drives (robot_hand/hand_retarget.py's flexion
mapping, same as scripts/hand_retarget_vedo.py). No arm/mocap, no MuJoCo --
just "is the Quest data arriving, and does it produce a sane robot hand".

Two data sources -- the Quest APK build is still in progress, so this
defaults to a source that needs no headset, no build, no network:
  mock  (default) -- in-process synthetic hand (devices/quest_hand/
    quest_hand_synth.py), fingers sweeping open/closed. Exercises the exact
    same xr_hand_to_state() math the real path uses, just skipping the wire.
  quest -- devices/quest_hand/quest_hand_reader.QuestHandUDPReader, the real
    receiver (see that module's docstring for the wire format and the
    XrHandsFB APK that has to be sending to it).

Known limitation: the skeleton is drawn in the device's own (Y-up) space,
recentered on the wrist but NOT rotated into the robot's Z-up frame -- so
its orientation relative to the URDF hand below it is not meaningful yet.
That alignment is wrist-orientation retargeting, still to come once this is
combined with the mocap arm IK (mirroring how robot_hand/upper_body_retarget
.py aligns Axis Studio's frame to the robot's). This script only answers
"is the trajectory arriving and is flexion sane" -- position/shape of the
skeleton, not its rotation.

Usage:
    python scripts/quest_hand_vedo.py                        # mock, no hardware
    python scripts/quest_hand_vedo.py --backend quest         # real headset
    python scripts/quest_hand_vedo.py --backend quest --quest-port 5005
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time
from typing import Dict, Optional

_ROOT = pathlib.Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402
import vedo  # noqa: E402

from devices.quest_hand.quest_hand_reader import (  # noqa: E402
    N_XR_JOINTS,
    XR_PALM,
    XR_WRIST,
    QuestHandUDPReader,
    _CHAIN,
    xr_hand_to_state,
)
from devices.quest_hand.quest_hand_synth import synthetic_xr_hand  # noqa: E402
from devices.senseglove.sg_types import FINGER_NAMES, HandState  # noqa: E402
from robot_hand.anydex_retarget import AnyDexDg5fRetargeter  # noqa: E402
from scripts.hand_retarget_vedo import (  # noqa: E402
    _HAND_COLOR,
    _JOINT_PREFIX,
    _MOUNT_ANCHOR,
    _URDF_PATHS,
    HandRig,
)

# Skeleton bone list: wrist-palm, wrist-to-each-finger-base, then consecutive
# joints within each finger's chain. Fixed for the process lifetime (same
# 26-joint layout every frame), which is what lets SkeletonRig update vedo's
# Lines/Points actors in place instead of rebuilding them (see
# scripts/mocap_skeleton_vedo.py's vertices= note this mirrors).
_EDGES = [(XR_WRIST, XR_PALM)]
for _chain in _CHAIN.values():
    _EDGES.append((XR_WRIST, _chain[0]))
    _EDGES.extend(zip(_chain, _chain[1:]))
_EDGE_START = np.array([a for a, _ in _EDGES])
_EDGE_END = np.array([b for _, b in _EDGES])

# Floats the skeleton above its side's URDF-hand anchor so the two don't
# overlap and are easy to tell apart on screen.
_SKELETON_OFFSET = np.array([0.0, 0.0, 0.25])
_FINGER_LABEL = {
    "thumb": "thumb",
    "index": "index",
    "middle": "middle",
    "ring": "ring",
    "pinky": "pinky",
}
_FINGER_ABBR = {
    "thumb": "T",
    "index": "I",
    "middle": "M",
    "ring": "R",
    "pinky": "P",
}
_LABEL_OFFSET = np.array([0.008, 0.006, 0.008])

_SKELETON_LABELS: Dict[int, str] = {
    XR_WRIST: "W",
    XR_PALM: "PALM",
}
for _finger_name, _chain in _CHAIN.items():
    _abbr = _FINGER_ABBR[_finger_name]
    for _local_idx, _xr_idx in enumerate(_chain):
        _SKELETON_LABELS[_xr_idx] = f"{_abbr}{_local_idx}"


def _bar(value: float, width: int = 12) -> str:
    n = int(round(float(np.clip(value, 0.0, 1.0)) * width))
    return "#" * n + "." * (width - n)


def _closing_limit(lower: float, upper: float) -> float:
    return upper if abs(upper) >= abs(lower) else lower


def _print_dg5f_dof_report(rigs: Dict[str, HandRig]) -> None:
    print("[quest_hand_vedo] DG5F finger DoF / retarget mapping")
    print("  rule: Quest skeleton joint geometry -> DG5F joints")
    print("  thumb: T1 -> dg_1_1..3 spherical approximation; T2 -> dg_1_4")
    print("  long fingers: F1 -> dg_N_1,dg_N_2; F2 -> dg_N_3; F3 -> dg_N_4")
    for side, rig in rigs.items():
        print(f"  {side}:")
        for finger in FINGER_NAMES:
            chain = rig.retargeter.chains[finger]
            print(f"    {_FINGER_LABEL[finger]} -> dg_{FINGER_NAMES.index(finger) + 1}")
            for idx, jname in enumerate(chain, start=1):
                joint = rig.tree.joints[jname]
                axis = " ".join(f"{x:g}" for x in joint.axis)
                role = "spread" if idx == 1 else "curl"
                close = 0.0 if idx == 1 else _closing_limit(joint.lower, joint.upper)
                print(
                    f"      {jname}: dof={role:9s} axis=({axis}) "
                    f"limit=[{joint.lower:+.3f},{joint.upper:+.3f}] close={close:+.3f}"
                )


def _print_skeleton_label_report() -> None:
    print("[quest_hand_vedo] OpenXR skeleton labels")
    print(f"  W={XR_WRIST}, PALM={XR_PALM}")
    for finger in FINGER_NAMES:
        labels = [
            f"{_SKELETON_LABELS[xr_idx]}={xr_idx}"
            for xr_idx in _CHAIN[finger]
        ]
        print(f"  {finger}: " + ", ".join(labels))


def _side_status(side: str, rig: HandRig, hs: Optional[HandState]) -> list[str]:
    if hs is None:
        return [f"{side}: no tracked sample yet"]

    lines = [f"{side}: tracked  source={hs.raw.get('source', 'unknown')}"]
    for i, finger in enumerate(FINGER_NAMES):
        flex = float(hs.flexion[i])
        spread = hs.raw.get("spread")
        spread_i = float(spread[i]) if spread is not None else 0.0
        info = rig.retargeter.last_clamp.get(finger)
        clamp = ""
        if info is not None and info.applied < info.requested - 1e-3:
            clamp = f" clamp {info.applied:.2f}/{info.requested:.2f}"

        chain = rig.retargeter.chains[finger]
        q1 = rig.last_angles.get(chain[0], 0.0)
        curl_angles = [rig.last_angles.get(j, 0.0) for j in chain[1:]]
        lines.append(
            f"{finger[:5]:5s} [{_bar(flex)}] {flex:4.2f} "
            f"spread={spread_i:+4.2f} q1={q1:+4.2f} "
            f"q2-4={curl_angles[0]:+4.2f},{curl_angles[1]:+4.2f},{curl_angles[2]:+4.2f}{clamp}"
        )
    return lines


class MockQuestHandSource:
    """In-process stand-in for QuestHandUDPReader -- no network, no APK.
    Same poll() -> (left, right) HandState contract, built from the same
    synthetic_xr_hand() + xr_hand_to_state() the wire-format self-check
    (scripts/quest_hand_selfcheck.py) and the fake UDP sender use, so this
    exercises the real math, just without the socket in between.
    """

    def __init__(self, period: float = 4.0):
        self._t0 = time.time()
        self._period = period
        self._poses = {
            "left": (np.eye(3), np.array([-0.2, 1.1, -0.4])),
            "right": (np.eye(3), np.array([0.2, 1.1, -0.4])),
        }

    def connect(self) -> None:
        self._t0 = time.time()

    def poll(self):
        t = time.time() - self._t0
        states = {}
        for side, phase in (("left", 0.0), ("right", 0.5)):
            curl = 0.5 - 0.5 * math.cos(2 * math.pi * (t / self._period + phase))
            rot, pos = self._poses[side]
            states[side] = xr_hand_to_state(side, synthetic_xr_hand(curl, rot, pos))
        return states["left"], states["right"]

    def close(self) -> None:
        pass


def _array_or_none(value):
    if value is None:
        return None
    return np.asarray(value, dtype=float).tolist()


class QuestHandRecorder:
    """JSONL recorder for the hand-retarget debug UI.

    Each frame stores the raw Quest/OpenXR skeleton, the compact HandState
    fields, the extracted human-angle features, and the DG5F joint angles
    that were actually shown by the UI. That makes recordings useful both
    for offline retarget tuning and for checking one suspicious live frame.
    """

    FORMAT = "quest_dg5f_hand_v1"

    def __init__(self, path: pathlib.Path, sides: list[str], backend: str, render: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.file = open(path, "w", encoding="utf-8")
        self.t0: Optional[float] = None
        self.frames_written = 0
        header = {
            "type": "header",
            "format": self.FORMAT,
            "backend": backend,
            "render": render,
            "sides": sides,
            "finger_names": list(FINGER_NAMES),
            "skeleton_labels": {str(k): v for k, v in sorted(_SKELETON_LABELS.items())},
        }
        json.dump(header, self.file)
        self.file.write("\n")
        self.file.flush()

    def write_frame(self, samples: Dict[str, HandState], rigs: Dict[str, HandRig]) -> None:
        now = time.monotonic()
        if self.t0 is None:
            self.t0 = now

        hands = {}
        for side, hs in samples.items():
            raw = hs.raw or {}
            rig = rigs[side]
            xr_joints = raw.get("xr_joints")
            human_angles = None
            if xr_joints is not None:
                try:
                    human_angles = rig.retargeter._quest_human_angles(xr_joints)
                except Exception as exc:
                    human_angles = {"error": str(exc)}
            hands[side] = {
                "timestamp": float(hs.timestamp),
                "source": raw.get("source", "unknown"),
                "flexion": _array_or_none(hs.flexion),
                "spread": _array_or_none(raw.get("spread")),
                "joint_positions": _array_or_none(hs.joint_positions),
                "xr_joints": _array_or_none(xr_joints),
                "human_angles": human_angles,
                "dg5f_angles": {name: float(value) for name, value in sorted(rig.last_angles.items())},
            }

        record = {
            "type": "frame",
            "t": now - self.t0,
            "wall_time": time.time(),
            "hands": hands,
        }
        json.dump(record, self.file)
        self.file.write("\n")
        self.file.flush()
        self.frames_written += 1

    def close(self) -> None:
        if not self.file.closed:
            self.file.close()
        print(f"[quest_hand_vedo] recorded {self.frames_written} frames -> {self.path}")


class SkeletonRig:
    """The raw 26-joint OpenXR hand as points + bone lines, recentered onto
    a fixed on-screen anchor (translate only -- see module docstring on why
    this doesn't rotate into the robot's frame yet). Actors are built lazily
    on the first real sample rather than in __init__ like HandRig's, since
    (unlike the URDF hand) there's no sane rest pose to show before any data
    has arrived.
    """

    def __init__(self, color: str, anchor: np.ndarray):
        self.color = color
        self.anchor = anchor
        self.joints_actor: Optional[vedo.Points] = None
        self.bones_actor: Optional[vedo.Lines] = None
        self.label_actors: Dict[int, vedo.Text3D] = {}

    def actors(self) -> list:
        actors = [a for a in (self.joints_actor, self.bones_actor) if a is not None]
        actors.extend(self.label_actors.values())
        return actors

    def update(self, xr_joints: np.ndarray, plotter: vedo.Plotter) -> None:
        wrist = xr_joints[XR_WRIST, :3]
        pts = xr_joints[:, :3] - wrist + self.anchor
        if self.joints_actor is None:
            self.joints_actor = vedo.Points(pts, r=8, c=self.color)
            plotter.add(self.joints_actor)
        else:
            self.joints_actor.vertices = pts

        starts = pts[_EDGE_START]
        ends = pts[_EDGE_END]
        if self.bones_actor is None:
            self.bones_actor = vedo.Lines(starts, ends, lw=3, c=self.color)
            plotter.add(self.bones_actor)
        else:
            interleaved = np.empty((2 * len(_EDGES), 3))
            interleaved[0::2] = starts
            interleaved[1::2] = ends
            self.bones_actor.vertices = interleaved

        for xr_idx, label in _SKELETON_LABELS.items():
            pos = pts[xr_idx] + _LABEL_OFFSET
            actor = self.label_actors.get(xr_idx)
            if actor is None:
                actor = vedo.Text3D(
                    label,
                    pos=pos,
                    s=0.012,
                    font="VictorMono",
                    justify="center",
                    c="black",
                    depth=0.0,
                    literal=True,
                )
                self.label_actors[xr_idx] = actor
                plotter.add(actor)
            else:
                actor.pos(pos)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backend", choices=("mock", "quest"), default="mock")
    parser.add_argument(
        "--retargeter", choices=("analytic", "anydex"), default="analytic",
        help="DG5F retarget backend: local joint-angle mapping or AnyDexRetarget optimizer",
    )
    parser.add_argument("--hand", choices=("left", "right", "both"), default="both")
    parser.add_argument("--render", choices=("mesh", "capsule"), default="capsule")
    parser.add_argument(
        "--quest-host", default="192.168.8.156",
        help="local address to bind -- the actual adapter's IP, not 0.0.0.0 (see "
             "devices/quest_hand/quest_hand_reader.py's QuestHandUDPReader docstring)",
    )
    parser.add_argument("--quest-port", type=int, default=5005)
    parser.add_argument("--hz", type=float, default=30.0, help="poll + retarget tick rate")
    parser.add_argument(
        "--render-hz", type=float, default=30.0,
        help="capped vedo redraw rate, decoupled from --hz (see hand_retarget_vedo.py)",
    )
    parser.add_argument(
        "--record", type=pathlib.Path, default=None,
        help="write shown Quest skeleton + DG5F retarget frames to this .jsonl file",
    )
    parser.add_argument(
        "--anydex-config-left", type=pathlib.Path, default=None,
        help="override AnyDex config for the left DG5F hand",
    )
    parser.add_argument(
        "--anydex-config-right", type=pathlib.Path, default=None,
        help="override AnyDex config for the right DG5F hand",
    )
    args = parser.parse_args()

    if args.backend == "quest":
        reader = QuestHandUDPReader(args.quest_host, args.quest_port)
    else:
        reader = MockQuestHandSource()
    reader.connect()

    sides = ["left", "right"] if args.hand == "both" else [args.hand]
    urdf_rigs = {side: HandRig(side, _HAND_COLOR[side], args.render) for side in sides}
    anydex_retargeters = {}
    if args.retargeter == "anydex":
        config_paths = {"left": args.anydex_config_left, "right": args.anydex_config_right}
        anydex_retargeters = {
            side: AnyDexDg5fRetargeter(side, config_paths[side])
            for side in sides
        }
        print("[quest_hand_vedo] retargeter=anydex (AnyDexRetarget KeyVectorOptimizer)")
        for side, rt in anydex_retargeters.items():
            print(f"  {side}: config={rt.config_path}")
            print(f"  {side}: qpos joints={', '.join(rt.joint_names)}")
    else:
        print("[quest_hand_vedo] retargeter=analytic (local joint-angle mapping)")
    _print_dg5f_dof_report(urdf_rigs)
    _print_skeleton_label_report()
    skeleton_rigs = {
        side: SkeletonRig(_HAND_COLOR[side], _MOUNT_ANCHOR[side] + _SKELETON_OFFSET)
        for side in sides
    }
    last_hand: Dict[str, HandState] = {}
    recorder = QuestHandRecorder(args.record, sides, args.backend, args.render) if args.record else None

    plotter = vedo.Plotter(
        title="Quest 3 hand tracking: raw skeleton + DG5F URDF (vedo)", bg="white", axes=4
    )
    plotter.add(vedo.Grid(s=(1, 1)).c("grey8").alpha(0.2))
    for rig in urdf_rigs.values():
        plotter.add(rig.actors())
    status_text = vedo.Text2D(
        "waiting for hand samples...",
        pos="top-left",
        s=0.72,
        font="VictorMono",
        c="black",
        bg="white",
        alpha=0.65,
    )
    plotter.add(status_text)

    render_interval = 1.0 / args.render_hz
    state = {
        "last_tick": None, "last_report": time.perf_counter(),
        "sum_compute": 0.0, "sum_render": 0.0, "n_since_report": 0,
        "last_render": 0.0, "n_rendered": 0,
    }

    def update(_evt):
        now = time.perf_counter()
        state["last_tick"] = now

        t0 = time.perf_counter()
        left, right = reader.poll()
        if left is not None:
            last_hand["left"] = left
        if right is not None:
            last_hand["right"] = right

        for side in sides:
            hs = last_hand.get(side)
            flexion = hs.flexion if hs is not None else np.zeros(5)
            if args.retargeter == "anydex" and hs is not None and "xr_joints" in hs.raw:
                angles = anydex_retargeters[side].retarget_openxr_joints(hs.raw["xr_joints"])
                urdf_rigs[side].update_actors_from_angles(angles)
            elif hs is not None and "xr_joints" in hs.raw:
                urdf_rigs[side].update_actors_from_openxr_joints(hs.raw["xr_joints"])
            elif hs is not None:
                urdf_rigs[side].update_actors_from_joint_positions(hs.joint_positions)
            else:
                urdf_rigs[side].update_actors(flexion)
            if hs is not None and "xr_joints" in hs.raw:
                skeleton_rigs[side].update(hs.raw["xr_joints"], plotter)
        if recorder is not None:
            visible_samples = {side: last_hand[side] for side in sides if side in last_hand}
            if visible_samples:
                recorder.write_frame(visible_samples, urdf_rigs)
        status_lines = [
            "Quest -> DG5F retarget monitor",
            "skeleton labels: W/PALM, T0..T3, I0..I4, M0..M4, R0..R4, P0..P4",
            "finger  flexion        value  spread q1(rad) q2,q3,q4(rad)",
        ]
        if recorder is not None:
            status_lines.append(f"REC {recorder.frames_written} frames -> {recorder.path}")
        for side in sides:
            status_lines.extend(_side_status(side, urdf_rigs[side], last_hand.get(side)))
        status_text.text("\n".join(status_lines))
        t1 = time.perf_counter()

        if t1 - state["last_render"] >= render_interval:
            plotter.render()
            state["last_render"] = t1
            state["n_rendered"] += 1
        t2 = time.perf_counter()

        state["sum_compute"] += t1 - t0
        state["sum_render"] += t2 - t1
        state["n_since_report"] += 1

        if now - state["last_report"] > 2.0:
            n = state["n_since_report"]
            achieved_hz = n / (now - state["last_report"])
            render_hz = state["n_rendered"] / (now - state["last_report"])
            tracked = ",".join(sorted(last_hand.keys())) or "none"
            print(
                f"[quest_hand_vedo] poll={achieved_hz:.1f} Hz (target={args.hz:.0f}) "
                f"render={render_hz:.1f} Hz (target={args.render_hz:.0f}) "
                f"compute={1000 * state['sum_compute'] / n:.2f} ms/frame "
                f"render_cost={1000 * state['sum_render'] / n:.2f} ms/frame "
                f"tracked={tracked}"
            )
            state["last_report"] = now
            state["sum_compute"] = state["sum_render"] = 0.0
            state["n_since_report"] = state["n_rendered"] = 0

    print(
        f"[quest_hand_vedo] backend={args.backend} retargeter={args.retargeter} "
        f"hand={args.hand} render={args.render} "
        f"record={args.record} -- close the window to stop"
    )
    try:
        plotter.add_callback("timer", update)
        plotter.timer_callback("create", dt=int(1000.0 / args.hz))
        plotter.show(interactive=True)
    finally:
        reader.close()
        if recorder is not None:
            recorder.close()
        plotter.close()


if __name__ == "__main__":
    main()
