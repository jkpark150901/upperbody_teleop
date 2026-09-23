"""Perception Neuron upper body -> RBY1 URDF retargeting demo in Vedo.

Calibrates instantly from the first landmark frame received -- no guided
pose sequence, no countdown -- using --calibration-pose (default:
attention) as the assumed robot reference. This assumes pose/body
calibration is already handled upstream in Axis Studio and the operator is
in a normal pose when this script starts; it does not duplicate that step.
Press R any time for another instant single-pose recalibration to the same
pose.

Multi-pose calibration (optional refinement, key-triggered, not timed):
press 1/2/3 any time to capture the *current* frame as attention/T-pose/
raise("만세", both arms straight up) into a buffer, then C to commit a
calibration fit jointly over whatever's been captured
(RBY1UpperBodyRetargeter.calibrate_multi()) -- pooling several poses gives
a better rotation fit than one snapshot, and also measures each arm
segment's true (fully-extended) length from whichever pose happens to
extend it most, turning on --bvh-scale-style segment-length scaling
automatically once committed (see calibrate_multi()'s docstring). 0 clears
the buffer without committing. An earlier version of this script tried a
guided, TIMED T-pose->attention(->reach) auto-sequence instead and found
it broke real tracking once a reference pose's angle turned out to be a
bad guess (see robot_hand/upper_body_retarget.py's _reference_q "raise"
case for how that was avoided this time -- checked against
forward_kinematics, not guessed) -- key-triggered capture avoids forcing
any particular pace/order on the operator.

NOTE: an earlier version of this script also had a step that tried to
trigger Axis Studio's own calibration via MocapApi (CommandCalibrateMotion)
-- removed on request, since pose calibration belongs to Axis Studio, not
here; see git history if revisiting it.

Usage:
    python scripts/upper_body_retarget_vedo.py --backend mock
    python scripts/upper_body_retarget_vedo.py --backend mocapapi --port 7002
    python scripts/upper_body_retarget_vedo.py --backend replay --replay-file demo.jsonl

    # Live mocap + SenseGlove together, recording both to one file:
    python scripts/upper_body_retarget_vedo.py --backend mocapapi --hand-backend bridge \
        --record recordings/session1.jsonl
    # Replay that recording, driving arms *and* fingers on the full URDF:
    python scripts/upper_body_retarget_vedo.py --backend replay --replay-file recordings/session1.jsonl

--record/--hand-backend are this script's own addition (CombinedRecorder /
CombinedReplayReader below) -- a separate, smaller format from
mocap_skeleton_vedo.py's full-skeleton recordings (this one stores the same
9 resolved UpperBodyLandmarks positions + head/wrist rotations this script
already computes every frame, plus per-side SenseGlove flexion, since
that's all RBY1UpperBodyRetargeter.update() + Dg5fHandRetargeter.retarget()
need to reproduce a session). --backend replay auto-detects which of the
two formats a file is (peeks at the header line) so old recordings such as
demo.jsonl keep working unchanged through ReplayLandmarkReader.
"""

from __future__ import annotations

import argparse
import bisect
import ctypes
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
_SDK = _ROOT / "MocapApi" / "demo" / "demo-py"
if str(_SDK) not in sys.path:
    sys.path.insert(0, str(_SDK))

import numpy as np  # noqa: E402
import vedo  # noqa: E402

from devices.quest_hand.quest_hand_reader import QuestHandUDPReader  # noqa: E402
from devices.senseglove.sg_reader import (  # noqa: E402
    MockSenseGloveReader,
    SenseGloveJSONBridgeReader,
    SenseGloveReaderBase,
    SGCoreSenseGloveReader,
)
from robot_hand.upper_body_retarget import (  # noqa: E402
    RBY1UpperBodyRetargeter,
    UpperBodyLandmarks,
)
from robot_hand import mesh_loader  # noqa: E402
from robot_hand.anydex_retarget import AnyDexDg5fRetargeter  # noqa: E402
from robot_hand.hand_retarget import Dg5fHandRetargeter  # noqa: E402
from robot_hand.urdf_fk import load_urdf  # noqa: E402
from teleop.articulation_protocol import ArticulationSender  # noqa: E402
from teleop.geometry import Pose  # noqa: E402
from teleop.geometry import quat_to_mat  # noqa: E402
from scripts.mocap_skeleton_vedo import ReplaySkeletonSource, build_hierarchy  # noqa: E402

_URDF = _ROOT / "rby1_dg5f.urdf"
_MOCAP_NAMES = {
    "hips": ("Hips",), "chest": ("Spine3", "Spine2", "Spine1", "Spine"), "head": ("Head",),
    "left_shoulder": ("LeftShoulder",), "left_elbow": ("LeftForeArm",), "left_wrist": ("LeftHand",),
    "right_shoulder": ("RightShoulder",), "right_elbow": ("RightForeArm",), "right_wrist": ("RightHand",),
}
_UPPER_LINKS = {
    "base", "link_torso_hp", "link_torso_5",
    *(f"link_left_arm_{i}" for i in range(7)),
    *(f"link_right_arm_{i}" for i in range(7)),
    "link_head_0", "link_head_1", "link_head_2",
}
_HAND_JOINT_PREFIX = {"left": "left_hand_lj_dg", "right": "right_hand_rj_dg"}
_HAND_LINK_PREFIX = {"left": "left_hand_", "right": "right_hand_"}
_HAND_COLOR = {"left": "gold", "right": "cyan3"}
# Dg5fHandRetargeter's self-collision check needs each finger link's
# <collision><capsule> geometry, which only scripts/extract_hand_urdf.py's
# standalone dg5f_{left,right}.urdf carry (see that script / hand_retarget.py)
# -- rby1_dg5f.urdf's own hand links only have their original mesh
# <collision>, so retargeting has to run against these standalone trees;
# their joint names match rby1_dg5f.urdf's 1:1 (same _HAND_JOINT_PREFIX),
# so the resulting angles merge straight into the combined tree's FK below.
_HAND_URDF_PATHS = {
    "left": _ROOT / "robot_hand" / "urdf" / "dg5f_left.urdf",
    "right": _ROOT / "robot_hand" / "urdf" / "dg5f_right.urdf",
}
class MocapLandmarkReader:
    def __init__(self, port: int, capture_skeleton: bool = False):
        from MocapApi import mocap_api as mcp
        self.mcp, self.port = mcp, port
        self.app = None
        self.avatar_handle = None
        self.last_latency_ms: Optional[float] = None
        # Off by default (an extra ~50 GetJointGlobalPosition calls/poll
        # for no benefit when nobody's watching the skeleton panel) --
        # --show-skeleton with --backend mocapapi turns this on so the
        # *same* live connection this reader already has can also feed a
        # raw full-skeleton comparison panel, instead of opening a second
        # MocapApi connection on the same port (mocap_skeleton_vedo.py's
        # SkeletonMonitor does that, but a second listener on one port
        # from the same process is untested and not worth the risk here).
        self.capture_skeleton = capture_skeleton
        self.names: Optional[list] = None
        self.edges: Optional[list] = None
        self.positions: dict = {}
        # Some MocapApi builds/streaming modes (live BVH UDP, at least)
        # return an error from get_avatar_posture_time() on every call --
        # and the SDK's own wrapper prints a debug line before raising, so
        # catching the exception alone still spams the console every poll.
        # Stop calling it after the first failure instead of retrying forever.
        self._posture_time_supported = True

    def connect(self):
        settings = self.mcp.MCPSettings()
        settings.set_udp(self.port)
        settings.set_bvh_data(self.mcp.MCPBvhData.Binary)
        settings.set_bvh_transformation(self.mcp.MCPBvhDisplacement.Enable)
        settings.set_bvh_rotation(self.mcp.MCPBvhRotation.YXZ)
        self.app = self.mcp.MCPApplication()
        self.app.set_settings(settings)
        ok, message = self.app.open()
        if not ok:
            raise RuntimeError(f"MocapApi open failed on UDP {self.port}: {message}")
        self.app.disable_event_cache()

    def _global_position(self, joint) -> np.ndarray:
        x, y, z = ctypes.c_float(), ctypes.c_float(), ctypes.c_float()
        err = joint.api.contents.GetJointGlobalPosition(
            ctypes.pointer(x), ctypes.pointer(y), ctypes.pointer(z), joint.handle
        )
        if err != self.mcp.MCPError.NoError:
            raise RuntimeError(f"GetJointGlobalPosition failed: {self.mcp.MCPError._fields[err]}")
        return np.array([x.value, y.value, z.value], dtype=float)

    def _global_rotation(self, joint) -> np.ndarray:
        """World-frame rotation matrix, NOT joint.get_local_rotation()
        (rotation relative to the PARENT joint -- confirmed against the
        SDK header: MocapApi.h declares a separate GetJointLocalRotation
        vs GetJointGlobalRotation, and the Python wrapper only exposes the
        local one, so this calls the C function directly via joint.api,
        same pattern as _global_position above).

        Using local rotation here was a real bug: comparing
        local(calibration) to local(now) via neutral.T @ current only
        measures a pure wrist-orientation change if the parent (forearm)
        frame hasn't moved between those two moments -- once the arm is
        posed away from the calibration reference, the delta is
        contaminated by however much the forearm itself has rotated,
        which is exactly why the wrist/hand looked fine near the
        calibration pose but swung to a wrong-looking orientation (readable
        as "not attached to the wrist") once the arm was raised.
        """
        x, y, z, w = ctypes.c_float(), ctypes.c_float(), ctypes.c_float(), ctypes.c_float()
        err = joint.api.contents.GetJointGlobalRotation(
            ctypes.pointer(x), ctypes.pointer(y), ctypes.pointer(z), ctypes.pointer(w), joint.handle
        )
        if err != self.mcp.MCPError.NoError:
            raise RuntimeError(f"GetJointGlobalRotation failed: {self.mcp.MCPError._fields[err]}")
        return quat_to_mat(np.array([w.value, x.value, y.value, z.value]))

    def _capture_latency_ms(self, avatar) -> Optional[float]:
        """Local wall clock minus the polled posture's device-side timestamp.

        Only meaningful when this process and the mocap PC share a clock
        (same machine, or NTP-synced) -- it's a diagnostic for *where* lag
        comes from (capture/BVH pipeline vs this viewer), not a certified
        hardware latency figure.
        """
        if not self._posture_time_supported:
            return None
        try:
            hour, minute, second, millisecond = avatar.get_avatar_posture_time()
        except Exception:
            self._posture_time_supported = False
            print(
                "[upper_body] get_avatar_posture_time() unsupported on this MocapApi "
                "stream (device-side capture latency will show N/A; poll/IK timing "
                "and track err are unaffected)"
            )
            return None
        device_ms = ((hour * 60 + minute) * 60 + second) * 1000 + millisecond
        local = time.localtime()
        local_ms = ((local.tm_hour * 60 + local.tm_min) * 60 + local.tm_sec) * 1000
        local_ms += int((time.time() % 1) * 1000)
        latency_ms = local_ms - device_ms
        # Wrap around midnight so a capture just before 00:00 vs a poll just
        # after it doesn't read as a ~24h latency spike.
        if latency_ms < -12 * 3600 * 1000:
            latency_ms += 24 * 3600 * 1000
        elif latency_ms > 12 * 3600 * 1000:
            latency_ms -= 24 * 3600 * 1000
        return float(latency_ms)

    def poll(self):
        # Drain fully rather than one poll_next_event() call per tick --
        # confirmed on real hardware (mocap_skeleton_vedo.py) that a single
        # call per tick can leave a persistent, non-shrinking backlog (Hz
        # looks perfectly healthy throughout, but every frame shown stays a
        # constant ~30s stale, because backlog inflow ends up matching the
        # rate this only ever chips away at it). Loop until truly empty.
        newest = False
        for _ in range(64):
            events = self.app.poll_next_event()
            if not events:
                break
            for event in events:
                if event.event_type == self.mcp.MCPEventType.AvatarUpdated:
                    self.avatar_handle = event.event_data.avatar_handle
                    newest = True
        if not newest or self.avatar_handle is None:
            return None
        avatar = self.mcp.MCPAvatar(self.avatar_handle)
        self.last_latency_ms = self._capture_latency_ms(avatar)
        joints = {j.get_name(): j for j in avatar.get_joints()}
        resolved = {
            key: next((name for name in candidates if name in joints), None)
            for key, candidates in _MOCAP_NAMES.items()
        }
        missing = [key for key, name in resolved.items() if name is None]
        if missing:
            raise RuntimeError(f"Mocap joints missing: {missing}; available={sorted(joints)}")
        values = {key: self._global_position(joints[name]) for key, name in resolved.items()}
        values["head_rotation"] = self._global_rotation(joints[resolved["head"]])
        # Feeds RBY1UpperBodyRetargeter's 3-DOF wrist-orientation IK block
        # (arm_4/5/6) -- must be world-frame (_global_rotation), not
        # get_local_rotation()'s parent-relative one; see that method's
        # docstring for the bug this replaced.
        values["left_wrist_rotation"] = self._global_rotation(joints[resolved["left_wrist"]])
        values["right_wrist_rotation"] = self._global_rotation(joints[resolved["right_wrist"]])
        if self.capture_skeleton:
            if self.names is None:
                self.names, self.edges = build_hierarchy(avatar)
            self.positions = {name: self._global_position(joint) for name, joint in joints.items()}
        return UpperBodyLandmarks(**values)

    def close(self):
        if self.app is not None:
            self.app.close()


class MockLandmarkReader:
    def __init__(self, tree):
        self.tree, self.t0 = tree, time.monotonic()
        # No device-side clock to compare against for synthetic data.
        self.last_latency_ms: Optional[float] = None

    def connect(self):
        self.t0 = time.monotonic()

    def poll(self):
        t = time.monotonic() - self.t0
        q = {
            "torso_hp": 0.12 * np.sin(0.6 * t), "torso_5": 0.2 * np.sin(0.35 * t),
            "left_arm_0": 0.45 * np.sin(0.7 * t), "left_arm_1": 0.55 + 0.35 * np.sin(0.5 * t),
            "left_arm_3": -0.65 - 0.35 * np.sin(0.8 * t),
            "right_arm_0": -0.45 * np.sin(0.7 * t), "right_arm_1": -0.55 - 0.35 * np.sin(0.5 * t),
            "right_arm_3": -0.65 - 0.35 * np.sin(0.8 * t + 1.0),
            "head_0": 0.25 * np.sin(0.4 * t), "head_1": 0.16 * np.sin(0.55 * t),
        }
        p = self.tree.forward_kinematics(q)
        # Small oscillating wrist roll so --backend mock also exercises the
        # 3-DOF wrist-orientation IK block, not just the position blocks.
        wrist_rot = quat_to_mat(np.array([
            np.cos(0.4 * t), 0.0, np.sin(0.4 * t), 0.0,
        ]))
        return UpperBodyLandmarks(
            hips=p["link_torso_hp"].pos, chest=p["link_torso_5"].pos, head=p["link_head_2"].pos,
            left_shoulder=p["link_left_arm_0"].pos, left_elbow=p["link_left_arm_3"].pos,
            left_wrist=p["link_left_arm_6"].pos, right_shoulder=p["link_right_arm_0"].pos,
            right_elbow=p["link_right_arm_3"].pos, right_wrist=p["link_right_arm_6"].pos,
            left_wrist_rotation=wrist_rot, right_wrist_rotation=wrist_rot,
        )

    def close(self):
        pass


class ReplayLandmarkReader:
    """Feeds a mocap_skeleton_vedo.py `--record` .jsonl file through this
    script's own retargeting pipeline -- same recording format and replay
    timing as that script's ReplaySkeletonSource, but resolves each frame's
    raw joint positions into an UpperBodyLandmarks via the same name
    fallback chains MocapLandmarkReader.poll() uses (_MOCAP_NAMES), so it's
    a drop-in third backend rather than a separate viewer.

    Recordings only ever capture positions (see mocap_skeleton_vedo.py's
    Recorder), never joint rotation, so head_rotation is always None here --
    the head follows the same position-based IK path as --backend mock.
    """

    def __init__(self, path: pathlib.Path, loop: bool = True):
        self.loop = loop
        self.last_latency_ms: Optional[float] = None
        records = []
        bad_lines = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    # An ungracefully-killed recording process can leave a
                    # torn last line -- skip it rather than losing the
                    # whole recording (see mocap_skeleton_vedo.py).
                    bad_lines += 1
                    continue
                if obj.get("type") == "frame":
                    pos = {k: np.array(v, dtype=float) for k, v in obj["pos"].items()}
                    records.append((obj["t"], pos))
        if bad_lines:
            print(f"[upper_body] WARNING: skipped {bad_lines} corrupt/unparseable line(s) in {path}")
        if not records:
            raise RuntimeError(f"No usable frames in recording: {path}")
        self._records = records
        self._times = [t for t, _ in records]
        self._duration = self._times[-1]
        self.t0: Optional[float] = None
        print(f"[upper_body] loaded recording: {len(records)} frames, {self._duration:.1f}s ({path})")

    def connect(self):
        self.t0 = time.monotonic()

    def poll(self):
        if self.t0 is None:
            self.connect()
        elapsed = time.monotonic() - self.t0
        if self._duration <= 0:
            t = 0.0
        elif self.loop:
            t = elapsed % self._duration
        else:
            t = min(elapsed, self._duration)
        idx = max(0, min(bisect.bisect_right(self._times, t) - 1, len(self._records) - 1))
        pos = self._records[idx][1]
        resolved = {
            key: next((name for name in candidates if name in pos), None)
            for key, candidates in _MOCAP_NAMES.items()
        }
        missing = [key for key, name in resolved.items() if name is None]
        if missing:
            raise RuntimeError(f"Recording missing joints: {missing}; available={sorted(pos)}")
        values = {key: pos[name] for key, name in resolved.items()}
        values["head_rotation"] = None
        return UpperBodyLandmarks(**values)

    def close(self):
        pass


def _mat_to_list(m: Optional[np.ndarray]):
    return None if m is None else np.asarray(m, dtype=float).tolist()


def _mat_from_list(v) -> Optional[np.ndarray]:
    return None if v is None else np.array(v, dtype=float)


def _hand_flexion(sample) -> Optional[np.ndarray]:
    if sample is None:
        return None
    return sample.flexion if hasattr(sample, "flexion") else sample


def _hand_spread(sample) -> Optional[np.ndarray]:
    if sample is None or not hasattr(sample, "raw"):
        return None
    return sample.raw.get("spread")


def _is_quest_hand(sample) -> bool:
    return bool(
        sample is not None
        and hasattr(sample, "joint_positions")
        and hasattr(sample, "raw")
        and sample.raw.get("source") == "quest_hand"
    )


class CombinedRecorder:
    """Writes UpperBodyLandmarks (+ optional per-side SenseGlove flexion)
    frames to a .jsonl file -- this script's own recording format, smaller
    and more targeted than mocap_skeleton_vedo.py's full raw-skeleton
    recordings: just the 9 resolved landmark positions + head/wrist
    rotations RBY1UpperBodyRetargeter.update() already computes every
    frame, plus per-side flexion for Dg5fHandRetargeter.retarget() --
    exactly what's needed to reproduce a session through both retargeters
    later via CombinedReplayReader, without re-resolving raw joint names.
    """

    FORMAT = "upper_body_combined_v1"
    _LANDMARK_KEYS = (
        "hips", "chest", "head", "left_shoulder", "left_elbow", "left_wrist",
        "right_shoulder", "right_elbow", "right_wrist",
    )

    def __init__(self, path: pathlib.Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(path, "w", encoding="utf-8")
        self.t0: Optional[float] = None
        json.dump({"type": "header", "format": self.FORMAT}, self.file)
        self.file.write("\n")
        self.file.flush()

    def write_frame(self, frame: UpperBodyLandmarks, hand: Dict[str, Optional[np.ndarray]]) -> None:
        now = time.monotonic()
        if self.t0 is None:
            self.t0 = now
        landmarks = {key: np.asarray(getattr(frame, key), dtype=float).tolist() for key in self._LANDMARK_KEYS}
        landmarks["head_rotation"] = _mat_to_list(frame.head_rotation)
        landmarks["left_wrist_rotation"] = _mat_to_list(frame.left_wrist_rotation)
        landmarks["right_wrist_rotation"] = _mat_to_list(frame.right_wrist_rotation)
        record = {
            "type": "frame",
            "t": now - self.t0,
            "landmarks": landmarks,
            "hand": {side: _mat_to_list(hand.get(side)) for side in ("left", "right")},
        }
        json.dump(record, self.file)
        self.file.write("\n")

    def close(self) -> None:
        self.file.close()


class CombinedReplayReader:
    """Replays a CombinedRecorder .jsonl -- feeds the arm IK
    (UpperBodyLandmarks, via poll()) and exposes the same frame's per-side
    SenseGlove flexion as .last_hand right after, so one --record session
    (live mocap + SenseGlove together) can drive the *whole* URDF -- arms
    and fingers -- back through one --backend replay, no separate hand
    backend needed on replay unless the caller wants to override it live.
    """

    def __init__(self, path: pathlib.Path, loop: bool = True):
        self.loop = loop
        self.last_latency_ms: Optional[float] = None
        self.last_hand: Dict[str, Optional[np.ndarray]] = {"left": None, "right": None}
        records = []
        bad_lines = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    bad_lines += 1
                    continue
                if obj.get("type") == "frame":
                    records.append(obj)
        if bad_lines:
            print(f"[upper_body] WARNING: skipped {bad_lines} corrupt/unparseable line(s) in {path}")
        if not records:
            raise RuntimeError(f"No usable frames in recording: {path}")
        self._records = records
        self._times = [r["t"] for r in records]
        self._duration = self._times[-1]
        self.t0: Optional[float] = None
        print(f"[upper_body] loaded combined recording: {len(records)} frames, {self._duration:.1f}s ({path})")

    def connect(self):
        self.t0 = time.monotonic()

    def poll(self) -> UpperBodyLandmarks:
        if self.t0 is None:
            self.connect()
        elapsed = time.monotonic() - self.t0
        if self._duration <= 0:
            t = 0.0
        elif self.loop:
            t = elapsed % self._duration
        else:
            t = min(elapsed, self._duration)
        idx = max(0, min(bisect.bisect_right(self._times, t) - 1, len(self._records) - 1))
        rec = self._records[idx]
        landmarks = rec["landmarks"]
        values = {key: np.array(landmarks[key], dtype=float) for key in CombinedRecorder._LANDMARK_KEYS}
        values["head_rotation"] = _mat_from_list(landmarks.get("head_rotation"))
        values["left_wrist_rotation"] = _mat_from_list(landmarks.get("left_wrist_rotation"))
        values["right_wrist_rotation"] = _mat_from_list(landmarks.get("right_wrist_rotation"))
        hand = rec.get("hand") or {}
        self.last_hand = {side: _mat_from_list(hand.get(side)) for side in ("left", "right")}
        return UpperBodyLandmarks(**values)

    def close(self):
        pass


def _replay_format(path: pathlib.Path) -> str:
    """Peek at a recording's header line to tell a CombinedRecorder file
    (this script's own --record) apart from a mocap_skeleton_vedo.py
    full-skeleton recording -- both are .jsonl with a "header" first line,
    just a different schema, and old recordings (e.g. demo.jsonl) need to
    keep replaying through ReplayLandmarkReader unchanged.
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                break
            if obj.get("type") == "header" and obj.get("format") == CombinedRecorder.FORMAT:
                return "combined"
            break
    return "skeleton"


def _matrix(pose: Pose) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = pose.as_matrix(), pose.pos
    return m


class LinkActor:
    def __init__(self, link, color):
        if link.visual_mesh:
            self.origin = link.visual_origin
            parts = []
            for mesh, rgba in mesh_loader.load_mesh_parts(link.visual_mesh):
                mesh.color(tuple(channel / 255.0 for channel in rgba[:3])).alpha(rgba[3] / 255.0)
                parts.append(mesh)
        elif link.collision_radius > 0:
            self.origin = link.collision_origin or Pose.identity()
            length, radius = link.collision_length, link.collision_radius
            parts = [vedo.Cylinder(r=radius, height=length, axis=(0, 0, 1), c=color),
                     vedo.Sphere(pos=(0, 0, -length / 2), r=radius, c=color),
                     vedo.Sphere(pos=(0, 0, length / 2), r=radius, c=color)]
        else:
            self.origin = Pose.identity()
            parts = [vedo.Sphere(r=0.035, c=color).alpha(0.65)]
        self.actor = vedo.Assembly(parts)

    def set_pose(self, pose):
        self.actor.transform.reset()
        self.actor.apply_transform(_matrix(pose.compose(self.origin)))


def _launch_calibration_gui(action_queue: "queue.Queue[str]", status_queue: "queue.Queue[str]") -> None:
    """Small Tk control panel for the multi-pose calibration buffer, in its
    own thread alongside the vedo/VTK window.

    vedo's own add_button()/Slider widgets raise AttributeError outright in
    this environment's VTK build (vtkOpenGLRenderer has no AddActor2D --
    same issue scripts/wrist_ik_test_vedo.py already hit and documented, so
    checked again here before relying on it: confirmed still broken), so a
    real on-screen GUI for this has to be a separate window rather than an
    in-scene widget. This thread only ever touches its own Tk widgets and
    the two queues -- all actual state (RBY1UpperBodyRetargeter.calibrate_multi,
    the vedo status_text actor, etc.) is mutated back on the main/VTK
    thread, inside update()'s per-tick action_queue drain, since VTK and Tk
    are each their own native GUI toolkit and neither is safe to poke at
    from a thread that isn't running its event loop.
    """
    import tkinter as tk

    root = tk.Tk()
    root.title("다중 자세 캘리브레이션")
    root.attributes("-topmost", True)

    tk.Label(
        root, text="전체 팔 관절 스케일 보정용 캘리브레이션", font=("Segoe UI", 10, "bold"), pady=6,
    ).pack()
    status_var = tk.StringVar(value="attention:X  tpose:X  raise:X")
    tk.Label(root, textvariable=status_var, font=("Consolas", 11), pady=4).pack()

    def send(action: str):
        return lambda: action_queue.put(action)

    capture_row = tk.Frame(root)
    capture_row.pack(pady=(2, 6))
    tk.Button(capture_row, text="차렷 캡처", width=12, command=send("capture:attention")).pack(side="left", padx=3)
    tk.Button(capture_row, text="T포즈 캡처", width=12, command=send("capture:tpose")).pack(side="left", padx=3)
    tk.Button(capture_row, text="만세 캡처", width=12, command=send("capture:raise")).pack(side="left", padx=3)

    action_row = tk.Frame(root)
    action_row.pack(pady=(0, 8))
    tk.Button(
        action_row, text="확정 (스케일 보정 ON)", width=20, bg="#dff0d8", command=send("commit")
    ).pack(side="left", padx=3)
    tk.Button(action_row, text="버퍼 초기화", width=12, command=send("clear")).pack(side="left", padx=3)
    tk.Button(
        action_row, text="즉시 단일 자세 재캘리브레이션", width=26, command=send("recalibrate")
    ).pack(side="left", padx=3)

    def poll_status():
        try:
            while True:
                status_var.set(status_queue.get_nowait())
        except queue.Empty:
            pass
        root.after(150, poll_status)

    root.after(150, poll_status)
    root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="PN -> RBY1 upper-body retargeting in Vedo")
    parser.add_argument("--backend", choices=("mock", "mocapapi", "replay"), default="mock")
    parser.add_argument("--port", type=int, default=7002)
    parser.add_argument(
        "--replay-file", type=pathlib.Path, default=None,
        help="--backend replay: a mocap_skeleton_vedo.py --record'ed .jsonl file",
    )
    parser.add_argument(
        "--no-loop", action="store_true",
        help="--backend replay: play once and hold on the last frame instead of looping",
    )
    parser.add_argument("--hz", type=float, default=30.0, help="poll + retarget tick rate")
    parser.add_argument(
        "--render-hz", type=float, default=30.0,
        help="capped vedo redraw rate, decoupled from --hz so a fast poll loop "
             "doesn't push every tick into the (expensive) VTK render call",
    )
    parser.add_argument("--calibration-pose", choices=("attention", "tpose", "raise"), default="attention")
    parser.add_argument(
        "--torso", choices=("fixed", "retarget"), default="fixed",
        help="keep torso at its initial pose (default) or include it in IK",
    )
    parser.add_argument(
        "--bvh-scale", action="store_true",
        help="EXPERIMENTAL: scale each arm/head segment's robot reach by how extended the "
             "human's own BVH segment currently is vs. its length at calibration time, instead "
             "of always using the robot's full segment length regardless of how bent the human's "
             "arm is. Off by default -- see RBY1UpperBodyRetargeter.use_bvh_scale.",
    )
    parser.add_argument(
        "--hand-backend", choices=("none", "mock", "sgcore", "bridge", "quest"), default="none",
        help="also drive the URDF's DG5F fingers from a SenseGlove reader, live alongside the "
             "arm mocap (see devices/senseglove/sg_reader.py -- same backends as "
             "hand_retarget_vedo.py), or from Quest 3 VR hand tracking streamed over UDP by a "
             "modified XrHandsFB APK ('quest', see devices/quest_hand/quest_hand_reader.py). "
             "'none' (default) leaves fingers at rest, unless a --backend replay recording has "
             "its own embedded flexion (see CombinedReplayReader).",
    )
    parser.add_argument(
        "--hand-retargeter", choices=("analytic", "anydex"), default="analytic",
        help="finger retarget backend for Quest hand samples. SenseGlove/mock flexion still uses analytic.",
    )
    parser.add_argument(
        "--anydex-config-left", type=pathlib.Path, default=None,
        help="override AnyDex config for the left DG5F hand",
    )
    parser.add_argument(
        "--anydex-config-right", type=pathlib.Path, default=None,
        help="override AnyDex config for the right DG5F hand",
    )
    parser.add_argument("--hand-bridge-host", default="127.0.0.1")
    parser.add_argument("--hand-bridge-port", type=int, default=8850)
    parser.add_argument(
        "--quest-host", default="192.168.8.156",
        help="local address to bind for the Quest's UDP hand-joint stream (--hand-backend quest) "
             "-- the actual adapter's IP, not 0.0.0.0 (see QuestHandUDPReader's docstring)",
    )
    parser.add_argument("--quest-port", type=int, default=5005)
    parser.add_argument(
        "--record", type=pathlib.Path, default=None,
        help="write every frame (landmarks + rotations + hand flexion, if any) to this .jsonl "
             "as it streams -- see CombinedRecorder. Works with any --backend.",
    )
    parser.add_argument(
        "--send", default=None, metavar="URL",
        help="stream the retargeted robot articulation (one radian value per rby1_dg5f.urdf "
             "revolute joint: torso, both arms, head, both hands' fingers) to a remote server, "
             "e.g. udp://192.168.0.20:9100 or tcp://192.168.0.20:9100 (this side connects; the "
             "server listens). Wire format: teleop/articulation_protocol.py; test receiver: "
             "scripts/articulation_receiver.py.",
    )
    parser.add_argument(
        "--send-hz", type=float, default=30.0,
        help="--send rate cap (never faster than the --hz poll/IK tick)",
    )
    parser.add_argument(
        "--show-skeleton", action="store_true",
        help="second panel with the raw human skeleton next to the robot, for visually checking "
             "retargeting against the source motion. Supported for --backend mocapapi (the same "
             "live connection also exposes every raw joint, no second MocapApi connection) and "
             "--backend replay of a mocap_skeleton_vedo.py-style full-skeleton recording; a "
             "combined recording (this script's own --record) or --backend mock have no "
             "independent raw skeleton to show, so this is a no-op there (with a warning).",
    )
    args = parser.parse_args()
    if args.backend == "replay" and args.replay_file is None:
        parser.error("--backend replay requires --replay-file")

    tree = load_urdf(str(_URDF))
    retargeter = RBY1UpperBodyRetargeter(tree, calibration_pose=args.calibration_pose)
    retargeter.use_bvh_scale = args.bvh_scale
    replay_kind = _replay_format(args.replay_file) if args.backend == "replay" else None
    if args.backend == "mock":
        reader = MockLandmarkReader(tree)
    elif args.backend == "replay":
        reader = (
            CombinedReplayReader(args.replay_file, loop=not args.no_loop) if replay_kind == "combined"
            else ReplayLandmarkReader(args.replay_file, loop=not args.no_loop)
        )
    else:
        reader = MocapLandmarkReader(args.port, capture_skeleton=args.show_skeleton)
    reader.connect()

    dual_panel = args.show_skeleton and (
        args.backend == "mocapapi" or (args.backend == "replay" and replay_kind == "skeleton")
    )
    if args.show_skeleton and not dual_panel:
        print(
            "[upper_body] --show-skeleton has no raw skeleton to show for "
            f"backend={args.backend}"
            + (f" (recording format={replay_kind})" if args.backend == "replay" else "")
            + " -- ignoring."
        )

    hand_reader: Optional[SenseGloveReaderBase] = None
    if args.hand_backend == "mock":
        hand_reader = MockSenseGloveReader()
    elif args.hand_backend == "sgcore":
        hand_reader = SGCoreSenseGloveReader()
    elif args.hand_backend == "bridge":
        hand_reader = SenseGloveJSONBridgeReader(args.hand_bridge_host, args.hand_bridge_port)
    elif args.hand_backend == "quest":
        hand_reader = QuestHandUDPReader(args.quest_host, args.quest_port)
    if hand_reader is not None:
        hand_reader.connect()
    hand_retargeters = {
        side: Dg5fHandRetargeter(load_urdf(str(_HAND_URDF_PATHS[side])), prefix)
        for side, prefix in _HAND_JOINT_PREFIX.items()
    }
    anydex_hand_retargeters = {}
    if args.hand_retargeter == "anydex":
        config_paths = {"left": args.anydex_config_left, "right": args.anydex_config_right}
        anydex_hand_retargeters = {
            side: AnyDexDg5fRetargeter(side, config_paths[side])
            for side in ("left", "right")
        }
        print("[upper_body] hand_retargeter=anydex (Quest samples only)")
        for side, rt in anydex_hand_retargeters.items():
            print(f"  {side}: config={rt.config_path}")
    last_hand: Dict[str, object] = {"left": None, "right": None}
    recorder = CombinedRecorder(args.record) if args.record is not None else None

    # Articulation stream: joint order = the IK's 18 torso/arm/head joints,
    # then each hand's 20 finger joints (thumb..pinky, _1.._4) -- the
    # receiver gets this list from the sender's "layout" message.
    hand_joint_names = [
        name for side in ("left", "right")
        for chain in hand_retargeters[side].chains.values() for name in chain
    ]
    stream_names = list(retargeter.names) + hand_joint_names
    sender = ArticulationSender(args.send, stream_names) if args.send else None
    send_interval = 1.0 / args.send_hz

    hand_links = {name for name in tree.links if name.startswith(tuple(_HAND_LINK_PREFIX.values()))}
    actors = {name: LinkActor(tree.links[name], "steelblue") for name in _UPPER_LINKS}
    for name in hand_links:
        side = "left" if name.startswith(_HAND_LINK_PREFIX["left"]) else "right"
        actors[name] = LinkActor(tree.links[name], _HAND_COLOR[side])
    initial_poses = retargeter.poses()
    for name, actor in actors.items():
        actor.set_pose(initial_poses[name])

    # --backend replay: two panels, one window (vedo/VTK's own
    # multi-renderer split, not two OS windows -- a real dual-window setup
    # needs two independent interactor event loops running at once in one
    # single-threaded process, which is a much bigger, harder-to-verify
    # undertaking). sharecam=False so each panel keeps its own camera --
    # rotating/zooming one no longer drags the other. An earlier version of
    # this used sharecam=True for a literal "same viewpoint" comparison,
    # but checking a rotation/mounting mismatch (e.g. roll/pitch/yaw
    # looking off, or a hand appearing offset from the wrist) needs
    # spinning each figure independently, not in lockstep.
    skeleton_source = None
    skeleton_state = {"joints_actor": None, "bones_actor": None}
    # MocapApi positions are centimeters (recordings/session1.jsonl, a
    # combined recording captured from the live reader, has hips y ~= 93),
    # live or recorded alike -- retargeting itself never notices because
    # _targets() only uses unit directions, but drawing next to a meters
    # robot needs the conversion. (An earlier version assumed live was
    # already meters.)
    SKELETON_SCALE = 0.01
    plotter = vedo.Plotter(
        shape=(1, 2) if dual_panel else (1, 1),
        title="Perception Neuron -> RBY1 upper body", bg="white", axes=4, sharecam=False,
    )
    plotter.at(0).add(vedo.Grid(s=(1.5, 1.5)).c("grey8").alpha(0.2))
    plotter.at(0).add([a.actor for a in actors.values()])
    status_text = vedo.Text2D("", pos="top-left", s=1.3, c="white", bg="red3", alpha=0.85, font="Calco")
    plotter.at(0).add(status_text)
    stats_text = vedo.Text2D("", pos="bottom-left", s=1.0, c="black", bg="white", alpha=0.75, font="Calco")
    plotter.at(0).add(stats_text)
    # Small markers at the IK's own targets for left/right wrist + head --
    # the 3 landmarks the IK has real authority over (shoulder/elbow always
    # carry some non-zero, non-IK-fixable error once torso is pinned, so
    # they're not a useful "is retargeting aiming right" signal). If a
    # marker visibly drifts to an anatomically wrong place as you move, the
    # bug is in retargeting math (_targets()/calibrate()), before IK ever
    # runs. If markers track correctly but the arm lags behind them, that's
    # the IK (see the on-screen "track err" number).
    target_markers = {key: vedo.Sphere(r=0.025, c="orange4").alpha(0.9) for key in ("left_wrist", "right_wrist", "head")}
    plotter.at(0).add(list(target_markers.values()))
    plotter.at(0).reset_camera()

    if dual_panel:
        if args.backend == "mocapapi":
            # No second MocapApi connection -- reader already captures the
            # full raw skeleton every poll() when capture_skeleton=True
            # (see MocapLandmarkReader), so it doubles as its own source
            # here; the render loop below skips re-polling it.
            skeleton_source = reader
            skeleton_label_text = "실시간 원본 스켈레톤 (raw mocap)"
        else:
            skeleton_source = ReplaySkeletonSource(args.replay_file, loop=not args.no_loop)
            skeleton_source.connect()
            skeleton_label_text = "녹화된 원본 스켈레톤 (cm->m 스케일)"
        skeleton_label = vedo.Text2D(
            skeleton_label_text, pos="top-left", s=1.1,
            c="black", bg="white", alpha=0.75, font="Calco",
        )
        plotter.at(1).add(skeleton_label)
    render_interval = 1.0 / args.render_hz
    state = {
        # True at startup -- calibrates instantly off the first frame that
        # arrives, no guided pose sequence (see module docstring: pose
        # calibration is Axis Studio's job, this just consumes the result).
        # Superseded any time a multi-pose calibration is committed (C
        # below) -- see _CALIB_POSE_KEYS/calib_buffer.
        "calibrate": True, "pose": args.calibration_pose, "last": time.monotonic(), "frames": 0,
        "last_render": 0.0,
        "latency_sum_ms": 0.0, "latency_n": 0,
        "ik_ms_sum": 0.0, "ik_ms_n": 0,
        "err_sum_m": 0.0, "err_n": 0,
        "last_frame": None,
        "tick": 0, "hand_tick": -1, "hand_joints": {}, "last_send": 0.0,
    }
    _POSE_LABEL = {"attention": "차렷", "tpose": "T-pose", "raise": "만세"}

    def recalibrate(pose=None):
        if pose is not None:
            state["pose"] = pose
        retargeter.set_calibration_pose(state["pose"])
        state["calibrate"] = True
        label = _POSE_LABEL[state["pose"]]
        status_text.text(f"{label} 즉시 초기화 -- 그 자세를 유지하세요").color("white").background("orange3")
        print(f"[upper_body] {label} initialization requested; hold that pose")

    # Multi-pose calibration buffer: 1/2/3 capture the *current* frame
    # (whatever update() below last saw) under that pose's name instead of
    # calibrating immediately, C commits calibrate_multi() over whatever's
    # been captured (1-3 poses; also turns on --bvh-scale-style segment
    # length scaling, since that's the whole point of capturing more than
    # one), 0 clears the buffer. An earlier version of this script tried a
    # timed auto-sequence through these same poses and found it broke real
    # tracking when guessed reference angles were wrong -- key-triggered
    # capture instead, on poses now checked against forward_kinematics
    # (see RBY1UpperBodyRetargeter._reference_q's "raise" case).
    _CALIB_POSE_KEYS = {"1": "attention", "2": "tpose", "3": "raise"}
    calib_buffer: Dict[str, UpperBodyLandmarks] = {}

    # GUI control panel (Tk, separate thread/window -- see
    # _launch_calibration_gui for why) alongside the existing 1/2/3/C/0/R
    # key bindings; both funnel through the same capture_pose/
    # commit_multi_calibration/clear_buffer functions below so status_text
    # and the Tk status label never disagree regardless of which one was
    # used.
    action_queue: "queue.Queue[str]" = queue.Queue()
    status_queue: "queue.Queue[str]" = queue.Queue()

    def _buffer_status() -> str:
        return " ".join(f"{p}{'✓' if p in calib_buffer else '✗'}" for p in ("attention", "tpose", "raise"))

    def capture_pose(pose: str) -> None:
        frame = state["last_frame"]
        if frame is None:
            print("[upper_body] no frame received yet -- can't capture")
            return
        calib_buffer[pose] = frame
        msg = f"{_POSE_LABEL[pose]} 캡처됨 [{_buffer_status()}] -- C=확정, 0=초기화"
        status_text.text(msg).color("white").background("blue3")
        status_queue.put(_buffer_status())
        print(f"[upper_body] captured '{pose}' into calibration buffer -- {_buffer_status()}")

    def commit_multi_calibration() -> None:
        if not calib_buffer:
            print("[upper_body] calibration buffer empty -- capture at least one pose first (1/2/3)")
            status_queue.put(f"{_buffer_status()} (버퍼 비어있음)")
            return
        active = "attention" if "attention" in calib_buffer else next(iter(calib_buffer))
        retargeter.calibrate_multi(calib_buffer, active_pose=active)
        retargeter.use_bvh_scale = True
        state["pose"] = active
        state["calibrate"] = False
        poses_used = "+".join(calib_buffer)
        status_text.text(
            f"다중 자세 캘리브레이션 완료 ({poses_used}) -- BVH 스케일 보정 ON"
        ).color("white").background("green4")
        status_queue.put(f"완료: {poses_used} (스케일 보정 ON)")
        print(f"[upper_body] calibrate_multi committed using poses: {poses_used}; use_bvh_scale=True")

    def clear_buffer() -> None:
        calib_buffer.clear()
        status_text.text("캘리브레이션 버퍼 초기화됨").color("white").background("orange3")
        status_queue.put(_buffer_status())
        print("[upper_body] calibration buffer cleared")

    gui_thread = threading.Thread(
        target=_launch_calibration_gui, args=(action_queue, status_queue), daemon=True
    )
    gui_thread.start()
    status_queue.put(_buffer_status())

    def dispatch_action(action: str) -> None:
        if action.startswith("capture:"):
            capture_pose(action.split(":", 1)[1])
        elif action == "commit":
            commit_multi_calibration()
        elif action == "clear":
            clear_buffer()
        elif action == "recalibrate":
            recalibrate()

    def current_hand_joints() -> Dict[str, float]:
        # Once per tick even if both --send and the render want it.
        if state["hand_tick"] != state["tick"]:
            angles: Dict[str, float] = {}
            for side, sample in last_hand.items():
                if _is_quest_hand(sample) and args.hand_retargeter == "anydex":
                    angles.update(anydex_hand_retargeters[side].retarget_openxr_joints(sample.raw["xr_joints"]))
                elif _is_quest_hand(sample):
                    angles.update(hand_retargeters[side].retarget_openxr_joints(sample.raw["xr_joints"]))
                else:
                    flexion = _hand_flexion(sample)
                    if flexion is not None:
                        angles.update(hand_retargeters[side].retarget(flexion, _hand_spread(sample)))
            state["hand_joints"], state["hand_tick"] = angles, state["tick"]
        return state["hand_joints"]

    def update(_event):
        nonlocal last_hand
        try:
            while True:
                dispatch_action(action_queue.get_nowait())
        except queue.Empty:
            pass
        now = time.monotonic()
        frame = reader.poll()
        if frame is None:
            return
        state["last_frame"] = frame
        if hand_reader is not None:
            left_hs, right_hs = hand_reader.poll()
            if left_hs is not None:
                last_hand["left"] = left_hs
            if right_hs is not None:
                last_hand["right"] = right_hs
        else:
            # --backend replay with a combined-format recording carries its
            # own per-frame flexion (see CombinedReplayReader) -- only used
            # when no *live* hand backend was asked for above.
            embedded = getattr(reader, "last_hand", None)
            if embedded is not None:
                last_hand = embedded
        if recorder is not None:
            recorder.write_frame(frame, {side: _hand_flexion(sample) for side, sample in last_hand.items()})
        if state["calibrate"]:
            retargeter.calibrate(frame)
            state["calibrate"] = False
            status_text.text(f"{_POSE_LABEL[state['pose']]} 초기화 완료 -- 로봇 제어 중").color("white").background("green4")
            print(
                f"[upper_body] {state['pose']} initialized -- 1/2/3=자세 캡처, C=다중 자세 확정, "
                "0=버퍼 초기화, R=즉시 단일 자세 재초기화"
            )
        # Timed separately from poll/FK/render so a "slow" complaint can be
        # pinned on the IK itself rather than guessed at from the overall
        # poll Hz (which also includes network/reader wait time).
        ik_t0 = time.perf_counter()
        joints = retargeter.update(frame, retarget_torso=args.torso == "retarget")
        state["tick"] += 1
        state["ik_ms_sum"] += (time.perf_counter() - ik_t0) * 1000.0
        state["ik_ms_n"] += 1
        # left_wrist/right_wrist/head are the landmarks IK has real
        # authority over (see target_markers comment above) -- averaging
        # just those gives a "is retargeting aiming where the IK can
        # actually follow" number, uncluttered by shoulder/elbow's
        # structural (torso-pinned) residual.
        tracked_err = [state and retargeter.last_error_m.get(k) for k in ("left_wrist", "right_wrist", "head")]
        tracked_err = [e for e in tracked_err if e is not None]
        if tracked_err:
            state["err_sum_m"] += sum(tracked_err) / len(tracked_err)
            state["err_n"] += 1
        # Full-tree FK (all ~75 joints, both hands' fingers included) and the
        # per-actor transform update are only meaningful right before an
        # actual redraw, so they're gated alongside plotter.render() —
        # otherwise they'd run at the poll tick rate for no visual benefit.
        # Same reasoning for Dg5fHandRetargeter.retarget() (it does a
        # capsule self-collision check, not free) -- only worth computing
        # for a frame that's actually about to be drawn.
        send_due = sender is not None and now - state["last_send"] >= send_interval
        render_due = now - state["last_render"] >= render_interval
        if send_due or render_due:
            hand_joints = current_hand_joints()
        if send_due:
            state["last_send"] = now
            sender.send(
                [{**joints, **hand_joints}.get(name, 0.0) for name in stream_names],
                time.time(),
                {side: flexion is not None for side, flexion in last_hand.items()},
            )
        if render_due:
            poses = tree.forward_kinematics({**joints, **hand_joints})
            for name, actor in actors.items():
                actor.set_pose(poses[name])
            for key, marker in target_markers.items():
                target_pos = retargeter.last_target.get(key)
                if target_pos is not None:
                    marker.pos(target_pos)
            if skeleton_source is not None:
                if skeleton_source is not reader:
                    # Live (mocapapi) case: reader.poll() above this tick
                    # already refreshed .names/.positions -- polling again
                    # here would just spend another live query for no
                    # reason (a replay source's own poll() is cheap/local).
                    skeleton_source.poll()
                if skeleton_source.names and skeleton_source.positions:
                    pts = SKELETON_SCALE * np.array(
                        [skeleton_source.positions.get(n, (0.0, 0.0, 0.0)) for n in skeleton_source.names]
                    )
                    if skeleton_state["joints_actor"] is None:
                        # Same vedo gotcha as mocap_skeleton_vedo.py: build
                        # at the real, final joint count once instead of
                        # growing a placeholder (.vertices = bigger_array
                        # only swaps point coordinates, never regenerates
                        # the underlying cell/topology array, so a grown
                        # actor silently caps at its construction-time size).
                        skeleton_state["joints_actor"] = vedo.Points(pts, r=8, c="green4")
                        plotter.at(1).add(skeleton_state["joints_actor"])
                    else:
                        skeleton_state["joints_actor"].vertices = pts
                    if skeleton_source.edges:
                        starts = SKELETON_SCALE * np.array(
                            [skeleton_source.positions.get(p, (0.0, 0.0, 0.0)) for p, _ in skeleton_source.edges]
                        )
                        ends = SKELETON_SCALE * np.array(
                            [skeleton_source.positions.get(c, (0.0, 0.0, 0.0)) for _, c in skeleton_source.edges]
                        )
                        if skeleton_state["bones_actor"] is None:
                            skeleton_state["bones_actor"] = vedo.Lines(starts, ends, lw=2, c="green4").alpha(0.6)
                            plotter.at(1).add(skeleton_state["bones_actor"])
                            # sharecam means this refits the ONE shared
                            # camera to panel 1's (fuller, full-body) bounds
                            # -- panel 0's smaller upper-body-only robot
                            # should still land comfortably inside that view.
                            plotter.at(1).reset_camera()
                        else:
                            interleaved = np.empty((2 * len(skeleton_source.edges), 3))
                            interleaved[0::2] = starts
                            interleaved[1::2] = ends
                            skeleton_state["bones_actor"].vertices = interleaved
            plotter.render()
            state["last_render"] = now
        state["frames"] += 1
        latency_ms = getattr(reader, "last_latency_ms", None)
        if latency_ms is not None:
            state["latency_sum_ms"] += latency_ms
            state["latency_n"] += 1
        if now - state["last"] >= 2.0:
            hz = state["frames"] / (now - state["last"])
            if state["latency_n"] > 0:
                latency_label = f"{state['latency_sum_ms'] / state['latency_n']:.1f} ms"
            else:
                # mock backend, or a MocapApi build without get_avatar_posture_time.
                latency_label = "N/A"
            ik_ms = state["ik_ms_sum"] / state["ik_ms_n"] if state["ik_ms_n"] else 0.0
            err_mm = (state["err_sum_m"] / state["err_n"]) * 1000.0 if state["err_n"] else 0.0
            print(
                f"[upper_body] {hz:.1f} Hz | IK {ik_ms:.1f} ms/frame | "
                f"capture->display latency {latency_label} | wrist/head track err {err_mm:.1f} mm"
            )
            stats_text.text(
                f"poll {hz:.1f} Hz   IK {ik_ms:.1f} ms   latency {latency_label}   track err {err_mm:.1f} mm"
            )
            state["last"], state["frames"] = now, 0
            state["latency_sum_ms"], state["latency_n"] = 0.0, 0
            state["ik_ms_sum"], state["ik_ms_n"] = 0.0, 0
            state["err_sum_m"], state["err_n"] = 0.0, 0
            state["latency_sum_ms"], state["latency_n"] = 0.0, 0

    def on_key(event):
        key = event.keypress.lower()
        if key in _CALIB_POSE_KEYS:
            capture_pose(_CALIB_POSE_KEYS[key])
        elif key == "c":
            commit_multi_calibration()
        elif key == "0":
            clear_buffer()
        elif key == "r":
            recalibrate()

    plotter.add_callback("key press", on_key)
    plotter.add_callback("timer", update)
    plotter.timer_callback("create", dt=max(1, int(1000 / args.hz)))
    print(f"[upper_body] backend={args.backend}, UDP={args.port}, torso={args.torso}, "
          f"hand_backend={args.hand_backend}, record={args.record}, send={args.send}, "
          f"poll={args.hz:.0f}Hz, render(capped)={args.render_hz:.0f}Hz, bvh_scale={args.bvh_scale}; "
          "keys: 1=차렷 2=T-pose 3=만세 캡처, C=다중 자세 캘리브레이션 확정(+BVH 스케일 보정 ON), "
          "0=버퍼 초기화, R=즉시 단일 자세 재캘리브레이션")
    try:
        plotter.show(interactive=True)
    finally:
        reader.close()
        if hand_reader is not None:
            hand_reader.close()
        if recorder is not None:
            recorder.close()
        if sender is not None:
            sender.close()
        plotter.close()


if __name__ == "__main__":
    main()
