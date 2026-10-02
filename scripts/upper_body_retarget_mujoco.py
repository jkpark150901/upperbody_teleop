"""PN/Quest -> RBY1 upper-body retargeting, driven live into MuJoCo.

Same retargeting pipeline as scripts/upper_body_retarget_vedo.py (same
backends/readers/retargeters, imported from there rather than duplicated)
but rendered in mujoco.viewer instead of vedo, as a kinematic puppet: every
tick writes the retargeted joint angles straight into qpos and calls
mj_forward() (poses only), never mj_step() -- there is no actuation,
gravity, or contact physics here, only "does the model's pose match the
retargeting output". A physics/actuator-driven version (mj_step with
position servos) is a natural next step once this is validated, not
something this script does.

Head is never driven (see --backend, --hand-backend etc. below): the real
robot this targets has no head DOF at all -- a physically fixed camera --
so head_0/head_1 are left at qpos 0 rather than fed calibrate()/update()'s
head-tracking output, and the mujoco.viewer camera is pointed once at a
fixed pose instead of the default orbit-follow behavior.

MuJoCo has no Collada (.dae) mesh decoder -- rby1_dg5f.urdf's visual meshes
are .dae (see robot_hand/mesh_loader.py's docstring) -- so this loads a
converted copy via robot_hand/mujoco_urdf.py (cached, regenerated only if
the source URDF changed) instead of rby1_dg5f.urdf directly.

Usage:
    python scripts/upper_body_retarget_mujoco.py --backend mock
    python scripts/upper_body_retarget_mujoco.py --backend quest \
        --quest-host 192.168.8.156 --quest-port 5005
    python scripts/upper_body_retarget_mujoco.py --backend mocapapi --port 7002
"""

from __future__ import annotations

import argparse
import queue
import pathlib
import sys
import threading
import time
from typing import Dict, Optional

_ROOT = pathlib.Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402
import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402

from devices.quest_hand.quest_hand_reader import QuestHandUDPReader  # noqa: E402
from devices.senseglove.sg_reader import (  # noqa: E402
    MockSenseGloveReader,
    SenseGloveJSONBridgeReader,
    SenseGloveReaderBase,
    SGCoreSenseGloveReader,
)
from devices.quest_hand.quest_hand_reader import XR_WRIST  # noqa: E402
from robot_hand.anydex_retarget import AnyDexDg5fRetargeter  # noqa: E402
from robot_hand.hand_retarget import Dg5fHandRetargeter  # noqa: E402
from robot_hand.mujoco_urdf import build_actuated_model, mujoco_compatible_urdf  # noqa: E402
from robot_hand.upper_body_retarget import (  # noqa: E402
    RBY1UpperBodyRetargeter, _HAND_MOUNT_ROTATION, _unit,
)
from robot_hand.urdf_fk import load_urdf  # noqa: E402
from scripts.upper_body_retarget_vedo import (  # noqa: E402
    _HAND_JOINT_PREFIX,
    _HAND_URDF_PATHS,
    _URDF,
    _hand_flexion,
    _hand_spread,
    _is_quest_hand,
    _replay_format,
    CombinedReplayReader,
    MockLandmarkReader,
    MocapLandmarkReader,
    ReplayLandmarkReader,
)

# The real robot has no head DOF (fixed camera) -- see module docstring.
# These never get written to; whatever qpos0 has for them is final.
_HEAD_JOINTS = ("head_0", "head_1")
_BVH_DIRECT_JOINTS = {
    "left": ("LeftArm", "LeftForeArm", "LeftHand"),
    "right": ("RightArm", "RightForeArm", "RightHand"),
}
_QUEST_TO_ROBOT = np.array([
    [0.0, 0.0, -1.0],
    [-1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
])  # fixed, hardware-confirmed mapping -- a per-session version derived from the captured wrists'
# left-right spread (replacing this) was tried and reverted: it broke hand ORIENTATION (confirmed
# on real hardware), since _QUEST_WRIST_FINGERS_LOCAL/_QUEST_WRIST_DORSAL_LOCAL were tuned against
# THIS fixed matrix specifically, not a session-varying one. Position/reach-direction instead gets
# its "where did the operator actually reach" fix from the head-pose-anchored offset in
# QuestForwardCalibration (base_handbase_head) -- the rotation path stays on this fixed matrix.

# OpenXR hand-tracking's wrist joint local axes -- confirmed on real
# hardware (same empirical process as robot_hand/upper_body_retarget.py's
# _BVH_HAND_FINGERS_LOCAL/_BVH_HAND_DORSAL_LOCAL, see bvh_upper_body_
# mapping.md), through two rounds of correction: fingers is local -Z (the
# rendered hand pointed exactly opposite the tracked hand's fingers with
# +Z), dorsal is local -Y (same symptom -- the rendered hand's back faced
# opposite the tracked hand's). Local +X stayed "right" (when the dorsal
# side faces up) throughout, unflipped. Same for both hands -- OpenXR
# already tracks left/right as separate skeletons, unlike BVH's single
# mirrored rig, so no left/right sign flip needed here the way
# _BVH_HAND_FINGERS_LOCAL has one.
_QUEST_WRIST_FINGERS_LOCAL = {"left": np.array([0.0, 0.0, -1.0]), "right": np.array([0.0, 0.0, -1.0])}
_QUEST_WRIST_DORSAL_LOCAL = np.array([0.0, -1.0, 0.0])

# Calibration scale safety margin (see QuestForwardCalibration.update()):
# the operator's calibrated max reach maps to this fraction of the
# robot's own max reach, not 100% of it, so reaching exactly as far as
# calibration doesn't land right on the joint-limit/reachability boundary.
_SCALE_SAFETY_MARGIN = 0.95

# link_{side}_arm_6 -> the DG5F hand's own base link, expressed in arm_6's
# own local frame: translate (0,0,-0.1087) then rotate 180deg about X
# (left/right_hand_mount's fixed joint), then translate (0,0,0.004) (the
# dg_base fixed joint, no further rotation) -- composed: a point at the
# hand base's origin sits at local (0,0,-0.1127) in arm_6's frame (same
# derivation and number as robot_hand/mujoco_urdf.py's wrist-camera
# placement). Used to convert between "arm_6 pose" and "hand base pose"
# in QuestForwardCalibration -- see its __init__/wrist_target.
_ARM6_TO_HANDBASE_OFFSET = np.array([0.0, 0.0, -0.1127])


def _quat_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(q, dtype=float)
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    return np.array([
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)],
    ])


class QuestForwardCalibration:
    """Forward-reach reference for --backend quest / hand_process.py's
    --arm-ik quest.

    A terminal command schedules capture a few seconds later, so the operator
    can type Enter, put both hands straight forward in view, and let that
    frame become the neutral pose. Wrist POSITION is tracked as a delta from
    that captured pose (remapped into the RBY1 frame), scaled PER ARM so the
    operator's own max reach at calibration maps to the robot's own max
    hand-base reach (robot_reach_limit, see _quest_reach_limits) -- not the
    earlier version's robot-vs-human WRIST SPAN (distance between the two
    captured wrists), which conflated "how far apart are my hands" with
    "how far can my arm reach" and had no principled relationship to
    whether the robot could actually cover the operator's range of motion.
    Wrist ORIENTATION is not delta-tracked at all: an earlier delta-from-
    calibration version here (conjugating a captured-relative rotation delta
    by _QUEST_TO_ROBOT, mirroring how robot_hand/upper_body_retarget.py's
    _wrist_rotation_target USED to work) turned out to never actually move
    the robot's wrist (confirmed not just slow/wrong but completely static
    regardless of hand rotation). _wrist_rotation_target was already
    rewritten once to fix the exact same failure mode for BVH/mocap (see its
    own docstring: that delta-conjugation approach "is only valid if the
    [source]'s own local rest frame happens to line up with [the alignment
    matrix]", which a one-time calibration capture doesn't guarantee) --
    wrist_target() below applies that same fix here: track the Quest wrist's
    current ABSOLUTE fingers/dorsal directions every frame (via
    _QUEST_WRIST_FINGERS_LOCAL/_QUEST_WRIST_DORSAL_LOCAL) instead of a delta
    from one captured frame.
    """

    def __init__(
        self, base_wrist: Dict[str, np.ndarray], base_rotation: Dict[str, np.ndarray],
        robot_reach_limit: Dict[str, float], head_pose: tuple[np.ndarray, np.ndarray],
        smoothing_factor: float = 1.0,
    ):
        self.base_wrist = base_wrist
        self.base_rotation = base_rotation
        self.robot_reach_limit = robot_reach_limit
        # Low-pass filter on the live wrist target (position delta, and the
        # fingers/dorsal direction vectors that orientation is built from),
        # ported from refer/HapticSyncController.cs's own smoothingFactor
        # (Vector3.Lerp / Quaternion.Slerp each frame -- "lower is
        # smoother"; 1.0 = no smoothing, matches this project's prior
        # behavior exactly). Reset per side on every (re)calibration (see
        # update()) so a new session doesn't start by smoothing toward a
        # stale direction from the last one.
        self.smoothing_factor = smoothing_factor
        self._filtered_delta_pos: Dict[str, np.ndarray] = {}
        self._filtered_fingers: Dict[str, np.ndarray] = {}
        self._filtered_dorsal: Dict[str, np.ndarray] = {}
        # The head camera's own fixed world pose (link_head_2 -- no head
        # DOF on this robot, see module docstring, so this never changes
        # once read) -- all wrist-target position math below runs in THIS
        # frame instead of the robot root's world frame, per request: a
        # reach target expressed relative to where the robot is actually
        # looking from is the more meaningful "origin" for a camera-driven
        # task than an arbitrary torso-attached world frame, and is also
        # what any future camera-relative training-data use would want.
        # Converted back to world only at the very end of wrist_target(),
        # since solve_wrist_pose()/the URDF tree's own FK still operate in
        # world/root frame regardless.
        self.head_pos, self.head_rot = head_pose
        # base_wrist/base_rotation (from _quest_forward_reference) are
        # link_{side}_arm_6's own pose -- the bare wrist joint, NOT where
        # the DG5F hand physically sits (confirmed: targeting arm_6
        # directly put the whole hand ~11cm further from the body than
        # intended, since that's the fixed arm_6->hand-base offset along
        # the forearm -- see robot_hand/mujoco_urdf.py's camera-placement
        # comment for the same 0.1127m number derived the same way).
        # wrist_target() needs to reason in HAND BASE space (the Quest
        # wrist marker corresponds to roughly where the user's hand base
        # is, not their forearm bone), then convert back to an arm_6
        # target at the end -- base_handbase is that space's own
        # calibration-pose reference point, computed once here.
        # p_handbase = p_arm6 + R_arm6 @ offset -- verified numerically via
        # direct FK (link_left_arm_6 vs left_hand_ll_dg_base), not just
        # derived by hand: offset's own value is exactly R_arm6.T @
        # (p_handbase - p_arm6).
        base_handbase_world = {
            side: base_wrist[side] + base_rotation[side] @ _ARM6_TO_HANDBASE_OFFSET
            for side in ("left", "right")
        }
        self.base_handbase_head = {
            side: self.head_rot.T @ (base_handbase_world[side] - self.head_pos)
            for side in ("left", "right")
        }
        self.quest_wrist: Dict[str, np.ndarray] = {}
        self.scale: Dict[str, float] = {"left": 1.0, "right": 1.0}
        self.ready = False
        self.capture_at: float | None = None
        self._last_prompt_t = 0.0

    @staticmethod
    def _wrist_pos(sample) -> np.ndarray:
        return np.asarray(sample.raw["xr_joints"][XR_WRIST, :3], dtype=float)

    @staticmethod
    def _wrist_rot(sample) -> np.ndarray:
        return _quat_xyzw_to_matrix(np.asarray(sample.raw["xr_joints"][XR_WRIST, 3:], dtype=float))

    def _filter_vec(self, state: Dict[str, np.ndarray], side: str, raw: np.ndarray) -> np.ndarray:
        """Lerp(state[side], raw, smoothing_factor), first-call passthrough
        (no prior value to blend from yet) -- the vector version of
        HapticSyncController.cs's Vector3.Lerp(filtered, raw, smoothingFactor)."""
        prev = state.get(side)
        filtered = raw.copy() if prev is None else prev + self.smoothing_factor * (raw - prev)
        state[side] = filtered
        return filtered

    def request_capture(self, delay_s: float = 3.0) -> None:
        self.capture_at = time.monotonic() + delay_s
        print(
            f"[quest_calibration] capture requested; extend both arms 90deg straight forward "
            f"with the backs of the hands up; capturing in {delay_s:.1f}s"
        )

    def update(self, last_hand: Dict[str, object], now: float) -> bool:
        if self.capture_at is None:
            # Was a one-shot print (self.prompted) -- confirmed the hard
            # way that's too easy to miss: this console window also prints
            # Quest connection stats on its own timer, which scrolls the
            # single calibration reminder off-screen within a few seconds,
            # so an operator who didn't catch it in that first window has
            # no later cue that nothing will happen until they type 'c'.
            # Repeats every 10s instead, until actually calibrated.
            if not self.ready and now - self._last_prompt_t > 10.0:
                self._last_prompt_t = now
                print(
                    "[quest_calibration] NOT YET CALIBRATED -- type 'c' or 'calibrate' then Enter "
                    "to capture forward-pose calibration after 3 seconds "
                    "(arms forward 90deg, backs of hands up). Nothing moves until this is done."
                )
            return self.ready

        remaining = self.capture_at - now
        if remaining > 0.0:
            return self.ready

        if not all(_is_quest_hand(last_hand.get(side)) for side in ("left", "right")):
            self.capture_at = None
            print(
                "[quest_calibration] skipped: both Quest hands must be tracked "
                "at capture time. Type 'c' to try again."
            )
            return self.ready

        self.quest_wrist = {side: self._wrist_pos(last_hand[side]).copy() for side in ("left", "right")}
        self._filtered_delta_pos = {}
        self._filtered_fingers = {}
        self._filtered_dorsal = {}
        # Per-arm scale: the operator's own reach at this capture -- the
        # straight-line (Euclidean) distance of the captured wrist pose's
        # own translation, i.e. norm(quest_wrist[side]) in OpenXR
        # LocalSpace -- mapped onto the robot's own max hand-base reach
        # (robot_reach_limit, see _quest_reach_limits), times a small
        # safety margin (_SCALE_SAFETY_MARGIN) so the mapped target stays
        # strictly inside the robot's reach even at the operator's exact
        # calibrated extension, not right on the boundary where closed-
        # form IK can still exist but joint-limit clipping starts biting.
        # e.g. operator reaches 1.0m, robot's own max is 0.5m -> scale
        # ~0.475, so the operator's own full extension maps to just inside
        # the robot's own full extension, same direction, just resized.
        #
        # LocalSpace's origin is an arbitrary anchor near wherever the
        # headset app started (see devices/quest_hand/quest_hand_reader.py's
        # own packet-format docstring) -- but that start-up pose error is
        # NOT something this needs to correct for: both the calibration
        # capture AND every live sample are read from the SAME LocalSpace,
        # so norm(quest_wrist[side]) consistently means "distance from
        # wherever the headset considers its own origin" either way, and
        # the live delta_pos_world below (wrist_pos - quest_wrist[side])
        # is already origin-independent (a pure difference) regardless.
        quest_reach = {side: float(np.linalg.norm(self.quest_wrist[side])) for side in ("left", "right")}
        self.scale = {
            side: float(_SCALE_SAFETY_MARGIN * self.robot_reach_limit[side] / quest_reach[side])
            if quest_reach[side] > 1e-6 else 1.0
            for side in ("left", "right")
        }
        self.ready = True
        self.capture_at = None
        print(
            "[quest_calibration] CALIBRATED: "
            + ", ".join(
                f"{side}: quest_reach={quest_reach[side]:.3f}m robot_reach={self.robot_reach_limit[side]:.3f}m "
                f"scale={self.scale[side]:.3f}"
                for side in ("left", "right")
            )
        )
        return True

    def wrist_target(self, side: str, sample, scale_multiplier: float) -> tuple[np.ndarray, np.ndarray]:
        """Position: delta from the captured calibration pose, tracked in
        HAND BASE space (base_handbase, see __init__) since that's what the
        Quest wrist marker conceptually corresponds to -- then converted
        back to an arm_6 target at the end (what solve_wrist_pose actually
        expects), using the offset rotated by THIS frame's own target
        orientation, not the calibration-time one (the offset direction
        rotates with the wrist). Orientation: the Quest wrist's current
        ABSOLUTE fingers/dorsal directions, transformed into robot-frame and
        composed with the URDF's fixed wrist-mount offset -- same structure
        as robot_hand/upper_body_retarget.py's _wrist_rotation_target, see
        this class's docstring for why NOT a delta from calibration. This
        (target_rot) is already arm_6's own intended orientation (it's built
        from _HAND_MOUNT_ROTATION the same way _wrist_rotation_target's
        result is matched against link_arm_6, not the hand itself), so it
        needs no further conversion -- only position does."""
        wrist_pos = self._wrist_pos(sample)
        delta_pos_world = _QUEST_TO_ROBOT @ (wrist_pos - self.quest_wrist[side])
        delta_pos_world = self._filter_vec(self._filtered_delta_pos, side, delta_pos_world)
        # Head-camera frame throughout (base_handbase_head, see __init__),
        # not robot-root world frame -- delta_pos_world is in world-AXIS
        # directions (from _QUEST_TO_ROBOT), so it still needs rotating
        # into the head's own frame before adding to a head-frame point;
        # converted back to world only at the very end, for solve_wrist_
        # pose()'s own FK (which is root/world-frame, unaffected by this).
        delta_pos_head = self.head_rot.T @ delta_pos_world
        target_pos_handbase_head = (
            self.base_handbase_head[side] + scale_multiplier * self.scale[side] * delta_pos_head
        )
        target_pos_handbase = self.head_pos + self.head_rot @ target_pos_handbase_head

        wrist_rot = self._wrist_rot(sample)
        fingers_world = _unit(_QUEST_TO_ROBOT @ (wrist_rot @ _QUEST_WRIST_FINGERS_LOCAL[side]))
        dorsal_world = _QUEST_TO_ROBOT @ (wrist_rot @ _QUEST_WRIST_DORSAL_LOCAL)
        # nlerp (normalized lerp), not slerp -- a cheaper approximation
        # that's fine at the small per-frame angle steps this smooths
        # (same role as HapticSyncController.cs's Quaternion.Slerp on the
        # wrist's rotation delta); re-orthogonalized below either way.
        fingers_world = _unit(self._filter_vec(self._filtered_fingers, side, fingers_world))
        dorsal_world = self._filter_vec(self._filtered_dorsal, side, dorsal_world)
        dorsal_world = _unit(dorsal_world - np.dot(dorsal_world, fingers_world) * fingers_world)
        third_world = np.cross(fingers_world, dorsal_world)  # local +Y, keeps (dorsal, +Y, fingers) right-handed
        palm_target = np.empty((3, 3))
        palm_target[:, 0] = dorsal_world
        palm_target[:, 1] = third_world
        palm_target[:, 2] = fingers_world
        target_rot = palm_target @ _HAND_MOUNT_ROTATION

        target_pos = target_pos_handbase - target_rot @ _ARM6_TO_HANDBASE_OFFSET
        return target_pos, target_rot


class MocapBvhJointAngleReader(MocapLandmarkReader):
    """Read BVH local joint rotations and map them directly to RBY1 qpos.

    This intentionally bypasses landmark retargeting and IK. Axis/MocapApi
    exposes BVH local Euler angles in the configured YXZ rotation mode; the
    RBY1 arm is not a BVH skeleton, so this is a direct axis mapping with
    clipping to the URDF joint limits, not a reach-preserving retargeter.
    """

    @staticmethod
    def _as_radians(euler) -> np.ndarray:
        values = np.asarray(euler, dtype=float)
        if np.max(np.abs(values)) > 2.0 * np.pi:
            values = np.deg2rad(values)
        return values

    def _local_euler(self, joint) -> np.ndarray:
        return self._as_radians(joint.get_local_rotation_by_euler())

    @staticmethod
    def _clamp(tree, name: str, value: float) -> float:
        joint = tree.joints[name]
        return float(np.clip(value, joint.lower, joint.upper))

    def poll_joint_angles(self, tree) -> Optional[Dict[str, float]]:
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

        joints = {j.get_name(): j for j in self.mcp.MCPAvatar(self.avatar_handle).get_joints()}
        missing = [
            bvh_name
            for names in _BVH_DIRECT_JOINTS.values()
            for bvh_name in names
            if bvh_name not in joints
        ]
        if missing:
            raise RuntimeError(f"Mocap BVH joints missing: {missing}; available={sorted(joints)}")

        # Applied as Axis Studio's own absolute local Euler values, not a
        # delta from a captured neutral -- simpler to reason about while
        # re-deriving the axis mapping, at the cost of assuming Axis
        # Studio's own BVH-zero pose already lines up with the robot's
        # relaxed (q=0) pose. If the robot idles in a visibly offset pose
        # when the performer is standing naturally, that assumption is
        # wrong and this needs to go back to a captured-neutral delta.
        raw = {
            bvh_name: self._local_euler(joints[bvh_name])
            for names in _BVH_DIRECT_JOINTS.values()
            for bvh_name in names
        }

        out: Dict[str, float] = {}
        for side in ("left", "right"):
            upper_name, forearm_name, hand_name = _BVH_DIRECT_JOINTS[side]
            upper = raw[upper_name]
            forearm = raw[forearm_name]
            hand = raw[hand_name]

            # Robot roles are fixed by the URDF's own joint axes (see
            # bvh_upper_body_mapping.md): arm_0 (~Y axis) is shoulder
            # flexion/extension, arm_1 (X axis) is abduction/adduction,
            # arm_2 (Z axis) is internal/external rotation. Re-derived from
            # real-hardware test reports (each round told us which BVH
            # index tracks which physical human motion, holding the robot
            # role fixed): index 0 of a bone's local Euler is that bone's
            # flexion-type component (sign opposite the robot's), index 1
            # is its roll/rotation-type component, index 2 is its
            # abduction-type component. Elbow/forearm-roll/wrist (arm_3..6)
            # extrapolate the same index-role pattern from the upper-arm-
            # bone result above; not yet independently confirmed. The right
            # shoulder lift joint has the opposite sign range from the
            # left, so mirror only that axis.
            lift_sign = 1.0 if side == "left" else -1.0
            mapped = {
                f"{side}_arm_0": upper[0],
                f"{side}_arm_1": lift_sign * upper[2],
                f"{side}_arm_2": upper[1],
                # Elbow flexion on RBY1 is negative from straight. Use the
                # BVH forearm bend magnitude so positive/negative Axis
                # conventions both bend the physical elbow the same way.
                f"{side}_arm_3": -abs(forearm[0]),
                # Forearm/wrist orientation: pass through remaining BVH
                # local rotations with clipping. These are intentionally
                # direct, not solved to match wrist pose.
                f"{side}_arm_4": forearm[1],
                f"{side}_arm_5": hand[0],
                f"{side}_arm_6": hand[2],
            }
            for name, value in mapped.items():
                out[name] = self._clamp(tree, name, value)
        return out


def _quest_forward_reference(retargeter: RBY1UpperBodyRetargeter) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Robot reference wrists for the Quest forward-reach calibration pose.

    Calibration pose convention: both arms are extended about 90 degrees
    straight forward, fingers point forward, and the back of each hand faces
    upward. In the DG5F palm frames, local +Z points toward the fingers and
    local +X is the hand normal used here as the dorsal/up axis.
    """
    retargeter.set_calibration_pose("attention")
    poses = retargeter.poses()
    for side in ("left", "right"):
        shoulder = poses[f"link_{side}_arm_0"].pos
        elbow = poses[f"link_{side}_arm_3"].pos
        wrist = poses[f"link_{side}_arm_6"].pos
        reach = 0.98 * (np.linalg.norm(elbow - shoulder) + np.linalg.norm(wrist - elbow))
        # Use the same closed-form arm position solve as live Quest tracking.
        # The old DLS-only solve_wrist() can settle into a low/downward local
        # minimum even for this reachable forward target, which makes the
        # post-calibration "arms forward" reference visibly droop.
        retargeter.solve_wrist_closed_form(side, shoulder + np.array([reach, 0.0, 0.0]))
        poses = retargeter.poses()

    desired_palm_rotation = np.array([
        [0.0, 0.0, 1.0],  # local +Z (fingers) -> robot +X (forward)
        [0.0, 1.0, 0.0],  # local +Y -> robot +Y (left)
        [-1.0, 0.0, 0.0],  # local +X (palm normal in this model) -> robot -Z (palm down)
    ])
    palm_link = {"left": "left_hand_ll_dg_palm", "right": "right_hand_rl_dg_palm"}
    for side in ("left", "right"):
        wrist_link = f"link_{side}_arm_6"
        wrist_to_palm = poses[wrist_link].as_matrix().T @ poses[palm_link[side]].as_matrix()
        target_wrist_rotation = desired_palm_rotation @ wrist_to_palm.T
        # Position is already solved above. Do only the wrist orientation
        # block here; solve_wrist_pose() would run position IK a second time
        # and can pick a different shoulder/elbow branch for the reference
        # pose, making the start/reset pose visibly asymmetric.
        retargeter._solve_wrist_orientation_block(side, target_wrist_rotation, retargeter.q.copy())
        poses = retargeter.poses()
    return (
        {side: poses[f"link_{side}_arm_6"].pos.copy() for side in ("left", "right")},
        {side: poses[f"link_{side}_arm_6"].as_matrix() for side in ("left", "right")},
    )


def _quest_reach_limits(
    retargeter: RBY1UpperBodyRetargeter,
    override_m: float | None = None,
) -> tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Shoulder centers and max wrist radii for Quest wrist targets --
    "wrist" here means the DG5F hand base (what wrist_target()'s targets
    actually are, see QuestForwardCalibration), so the straight-arm max
    distance includes the fixed arm_6->hand-base offset
    (_ARM6_TO_HANDBASE_OFFSET), not just the URDF's own upper+lower arm
    segment lengths -- leaving it out previously meant both the live
    reach-limit gate and the calibration scale (see QuestForwardCalibration.
    update()) were using a ~11cm-short max reach number."""
    poses = retargeter.poses()
    shoulders = {side: poses[f"link_{side}_arm_0"].pos.copy() for side in ("left", "right")}
    limits = {}
    handbase_offset_mag = float(np.linalg.norm(_ARM6_TO_HANDBASE_OFFSET))
    for side in ("left", "right"):
        upper = np.linalg.norm(retargeter.tree.joints[f"{side}_arm_3"].origin.pos)
        lower = np.linalg.norm(retargeter.tree.joints[f"{side}_arm_4"].origin.pos)
        max_reach = upper + lower + handbase_offset_mag
        limits[side] = float(override_m if override_m is not None else 0.98 * max_reach)
    return shoulders, limits


def _start_quest_calibration_console(commands: "queue.SimpleQueue[float]") -> threading.Thread:
    def run() -> None:
        print("[upper_body_mujoco] console commands: c/calibrate = capture forward pose after 3 seconds")
        for line in sys.stdin:
            cmd = line.strip().lower()
            if cmd in {"c", "calibrate", "capture", "r", "recalibrate"}:
                commands.put(3.0)
            elif cmd:
                print("[upper_body_mujoco] unknown command; use 'c' or 'calibrate'")

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


class MujocoJointWriter:
    """Joint-name -> qpos index, and a write() that's a no-op for any name
    not in the model (lets the same retarget joint dict this repo already
    produces -- 18 arm/torso/head + 20+20 finger names -- drive a MuJoCo
    model without either side needing to special-case the other)."""

    def __init__(self, model: mujoco.MjModel, skip: tuple = ()):
        self.model = model
        self.skip = set(skip)
        self._qposadr: Dict[str, int] = {}
        for i in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name:
                self._qposadr[name] = model.jnt_qposadr[i]
        # Only present when the model came from build_actuated_model()
        # (--sim-mode physics) -- act_{joint name}, see that function.
        self._ctrladr: Dict[str, int] = {}
        for i in range(model.nu):
            act_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if act_name and act_name.startswith("act_"):
                self._ctrladr[act_name[len("act_"):]] = i

    def write(self, data: mujoco.MjData, joints: Dict[str, float]) -> int:
        """Writes straight to qpos (kinematic puppet -- see module
        docstring). Returns how many names in `joints` had no matching
        MuJoCo joint (expected to be 0 once the model/retargeter names
        agree; printed once by the caller if non-zero, not spammed every
        tick)."""
        missing = 0
        for name, value in joints.items():
            if name in self.skip:
                continue
            adr = self._qposadr.get(name)
            if adr is None:
                missing += 1
                continue
            data.qpos[adr] = value
        return missing

    def write_ctrl(self, data: mujoco.MjData, joints: Dict[str, float]) -> int:
        """Writes to each joint's position-actuator ctrl instead of qpos
        directly (--sim-mode physics) -- a servo target for mj_step to
        converge toward under real gravity/contact dynamics, not the
        literal instantaneous pose. Same missing-name accounting as
        write()."""
        missing = 0
        for name, value in joints.items():
            if name in self.skip:
                continue
            adr = self._ctrladr.get(name)
            if adr is None:
                missing += 1
                continue
            data.ctrl[adr] = value
        return missing


class SelfCollisionMonitor:
    """Flags MuJoCo contacts not already present at the robot's own
    neutral/rest pose, after each mj_forward.

    Not every contact MuJoCo reports is a real self-collision: at rest,
    each arm's forearm/wrist collision capsules already overlap that same
    arm's own fingers by design (loose-fitting capsules approximating the
    real mesh, not an exact fit) -- confirmed ~109 such contacts already
    present at the model's own qpos==0 pose. An earlier version of this
    filtered by limb (left/right/torso) instead, but that also silently
    hid a genuine same-arm fold-back collision (e.g. the upper arm folding
    back into its own wrist) in testing, since that pair is still
    "same-limb". Diffing against the exact baseline pair set instead
    catches that while still suppressing the persistent capsule-overlap
    noise, regardless of which limb(s) are involved.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, warn_interval: float = 1.0):
        self.model = model
        self.warn_interval = warn_interval
        self._last_warn = 0.0
        self.last_pairs: set = set()
        self._baseline = self._raw_pairs(data)

    def _raw_pairs(self, data: mujoco.MjData) -> set:
        pairs = set()
        for i in range(data.ncon):
            c = data.contact[i]
            # Flex (cloth) contacts report geom1/geom2 as -1 (no rigid geom
            # on that side) -- model.geom_bodyid[-1] is a *valid* Python
            # index (wraps to the last element, some unrelated body), not
            # an error, so this silently produced bogus same-body pairs
            # like ('table', 'table') instead of raising. Skip anything
            # that isn't a real geom-geom contact; this monitor is about
            # the robot's own rigid links, not flex self-contact.
            if c.geom1 < 0 or c.geom2 < 0:
                continue
            b1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, self.model.geom_bodyid[c.geom1])
            b2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, self.model.geom_bodyid[c.geom2])
            if b1 is None or b2 is None:
                continue
            pairs.add(tuple(sorted((b1, b2))))
        return pairs

    def check(self, data: mujoco.MjData, now: float, warn: bool = True) -> set:
        pairs = self._raw_pairs(data) - self._baseline
        self.last_pairs = pairs
        if warn and pairs and now - self._last_warn >= self.warn_interval:
            self._last_warn = now
            print(f"[upper_body_mujoco] SELF-COLLISION: {sorted(pairs)}")
        return pairs


class LandmarkPanel:
    """Sends raw human landmarks (hips/chest/head/shoulder/elbow/wrist) over
    loopback UDP to scripts/landmark_graph_viewer.py, spawned as a SEPARATE
    PROCESS (auto-spawned here via subprocess.Popen).

    Deliberately not in-process: an in-process vedo (VTK/OpenGL) window
    crashed alongside mujoco.viewer's own GLFW-based passive viewer, and
    the requested replacement (Open3D, for free rotation -- matplotlib
    alone can't survive ax.cla() every frame without resetting the view
    angle back to default) also owns a GLFW/OpenGL context, so it carries
    the same same-process crash risk. A tiny UDP JSON payload is the whole
    IPC surface, keeping that GL context in its own process.

    Sends exactly the values UpperBodyLandmarks.as_dict() feeds into
    RBY1UpperBodyRetargeter._targets() -- i.e. what the tracker actually
    captured, before Kabsch alignment/--bvh-scale. Lets you tell "IK solved
    the wrong pose" apart from "the robot pose is a faithful (if extreme)
    copy of what the human/BVH data itself looks like right now" at a
    glance, instead of guessing from the MuJoCo view alone.
    """

    _POINTS = (
        "hips", "chest", "head",
        "left_shoulder", "left_elbow", "left_wrist",
        "right_shoulder", "right_elbow", "right_wrist",
    )

    def __init__(self, port: int = 5099):
        import socket
        import subprocess
        import sys
        self.port = port
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "scripts.landmark_graph_viewer", "--port", str(port)],
            cwd=str(_ROOT),
        )

    def update(self, human) -> None:
        import json
        h = human.as_dict()
        payload = json.dumps({
            "points": {name: h[name].tolist() for name in self._POINTS},
            # Wrist/head orientation IK already uses these (see
            # RBY1UpperBodyRetargeter._wrist_rotation_target()) -- sent here
            # too so the viewer can draw axis triads and confirm visually
            # that tracked orientation looks right, not just position.
            "rotation": {
                "left_wrist": None if human.left_wrist_rotation is None else np.asarray(human.left_wrist_rotation).tolist(),
                "right_wrist": None if human.right_wrist_rotation is None else np.asarray(human.right_wrist_rotation).tolist(),
                "head": None if human.head_rotation is None else np.asarray(human.head_rotation).tolist(),
            },
        }).encode("utf-8")
        self._sock.sendto(payload, ("127.0.0.1", self.port))

    def close(self) -> None:
        self._sock.close()
        self._proc.terminate()


def main() -> None:
    parser = argparse.ArgumentParser(description="PN/Quest -> RBY1 upper-body retargeting in MuJoCo")
    parser.add_argument("--backend", choices=("mock", "mocapapi", "replay", "quest"), default="mock")
    parser.add_argument("--port", type=int, default=7002)
    parser.add_argument("--replay-file", type=pathlib.Path, default=None)
    parser.add_argument("--no-loop", action="store_true")
    parser.add_argument("--hz", type=float, default=30.0, help="poll + retarget + sim-sync tick rate")
    parser.add_argument(
        "--mocap-drive", choices=("ik", "bvh-joints"), default="ik",
        help="--backend mocapapi only: 'ik' uses landmark IK with BVH local-angle hints for "
             "branch direction; 'bvh-joints' writes Axis BVH local joint angles directly to "
             "the robot arm joints without IK.",
    )
    parser.add_argument(
        "--calibration-pose", choices=("attention", "tpose", "raise"), default="attention",
        help="mocap/replay calibration pose. --backend quest uses terminal-triggered forward-pose calibration.",
    )
    parser.add_argument("--torso", choices=("fixed", "retarget"), default="fixed")
    parser.add_argument(
        "--sim-mode", choices=("physics", "kinematic"), default="physics",
        help="physics (default): build_actuated_model()'s position-actuated model, driven with "
        "mj_step so gravity/contacts are real -- the retargeted joint angles become actuator "
        "ctrl targets (a servo target), not the literal pose. kinematic: the old mj_forward-only "
        "puppet (writes qpos directly, no dynamics) -- useful for judging the IK/retargeting "
        "output on its own, without physics settling error or actuator gain tuning in the way.",
    )
    parser.add_argument(
        "--arm-solver", choices=("closed-form", "mink"), default="closed-form",
        help="closed-form: RBY1UpperBodyRetargeter's own ZXZ/ZYZ branch-search IK. "
        "mink: reuse the same calibration/targets but solve elbow+wrist as a mink QP "
        "task instead (requires `pip install mink`). mocapapi/mock/replay backends only.",
    )
    parser.add_argument(
        "--show-landmarks", action="store_true",
        help="open a second window plotting the raw human landmarks (hips/chest/head/"
        "shoulder/elbow/wrist) this tick's IK target came from, so an overextended/bent-"
        "wrong robot pose in the MuJoCo view can be told apart from IK actually being wrong "
        "vs. just following what the tracker captured. mocapapi/mock/replay backends only "
        "(quest and --mocap-drive bvh-joints have no UpperBodyLandmarks frame to plot).",
    )
    parser.add_argument(
        "--hand-backend", choices=("none", "mock", "sgcore", "bridge", "quest"), default="none",
        help="hand source for non-Quest arm backends. --backend quest automatically uses quest.",
    )
    parser.add_argument(
        "--hand-retargeter", choices=("analytic", "anydex"), default="anydex",
        help="finger retarget backend for Quest hand samples (see upper_body_retarget_vedo.py)",
    )
    parser.add_argument("--anydex-config-left", type=pathlib.Path, default=None)
    parser.add_argument("--anydex-config-right", type=pathlib.Path, default=None)
    parser.add_argument("--hand-bridge-host", default="127.0.0.1")
    parser.add_argument("--hand-bridge-port", type=int, default=8850)
    parser.add_argument("--quest-host", default="192.168.8.156")
    parser.add_argument("--quest-port", type=int, default=5005)
    parser.add_argument(
        "--wrist-scale", type=float, default=1.0,
        help="--backend quest only: see upper_body_retarget_vedo.py's --wrist-scale",
    )
    parser.add_argument(
        "--quest-max-reach", type=float, default=None,
        help="--backend quest only: max shoulder-to-wrist target radius in metres. "
             "Default is 98%% of the URDF upper+lower arm length. Targets beyond this are clamped "
             "onto that radius, same direction (ported from refer/HapticSyncController.cs).",
    )
    parser.add_argument(
        "--wrist-smoothing", type=float, default=1.0,
        help="--backend quest only: low-pass filter factor on the live wrist target, ported from "
        "refer/HapticSyncController.cs's smoothingFactor. 1.0 = no smoothing (default).",
    )
    args = parser.parse_args()
    if args.backend == "replay" and args.replay_file is None:
        parser.error("--backend replay requires --replay-file")
    if args.backend == "quest":
        if args.hand_backend not in ("none", "quest"):
            parser.error("--backend quest uses the Quest hand stream directly; do not pass another --hand-backend")
        args.hand_backend = "quest"
    if args.backend != "mocapapi" and args.mocap_drive != "ik":
        parser.error("--mocap-drive is only meaningful with --backend mocapapi")

    if args.sim_mode == "physics":
        model = build_actuated_model(_URDF)
    else:
        mjcf_urdf = mujoco_compatible_urdf(_URDF)
        model = mujoco.MjModel.from_xml_path(str(mjcf_urdf))
    data = mujoco.MjData(model)
    writer = MujocoJointWriter(model, skip=_HEAD_JOINTS)
    mujoco.mj_forward(model, data)  # baseline pose (qpos==0) for SelfCollisionMonitor's exclude set
    collision_monitor = SelfCollisionMonitor(model, data)

    tree = load_urdf(str(_URDF))  # this repo's own FK/IK tree, not MuJoCo's -- retargeting math runs on this
    retargeter = RBY1UpperBodyRetargeter(tree, calibration_pose=args.calibration_pose)
    mink_ik = None
    if args.arm_solver == "mink":
        from robot_hand.mink_upper_body_retarget import MinkUpperBodyIK  # optional dep, see that module
        mink_ik = MinkUpperBodyIK(retargeter, _URDF)

    landmark_panel = None
    if args.show_landmarks:
        if args.backend == "quest" or args.mocap_drive == "bvh-joints":
            parser.error("--show-landmarks needs a backend that produces UpperBodyLandmarks "
                         "(mocapapi with --mocap-drive ik, mock, or replay)")
        landmark_panel = LandmarkPanel()

    reader = None
    base_wrist: Dict[str, np.ndarray] = {}
    base_wrist_rotation: Dict[str, np.ndarray] = {}
    quest_shoulder: Dict[str, np.ndarray] = {}
    quest_reach_limit: Dict[str, float] = {}
    quest_calibration: Optional[QuestForwardCalibration] = None
    quest_calibration_commands: Optional["queue.SimpleQueue[float]"] = None
    if args.backend == "quest":
        base_wrist, base_wrist_rotation = _quest_forward_reference(retargeter)
        quest_shoulder, quest_reach_limit = _quest_reach_limits(retargeter, args.quest_max_reach)
        head_pose_ref = retargeter.poses()["link_head_2"]
        head_pose = (head_pose_ref.pos.copy(), head_pose_ref.as_matrix())
        quest_calibration = QuestForwardCalibration(
            base_wrist, base_wrist_rotation, quest_reach_limit, head_pose, args.wrist_smoothing,
        )
        quest_calibration_commands = queue.SimpleQueue()
        _start_quest_calibration_console(quest_calibration_commands)
        print(
            "[upper_body_mujoco] Quest reach limits: "
            + ", ".join(f"{side}={quest_reach_limit[side]:.3f}m" for side in ("left", "right"))
        )
    elif args.backend == "mock":
        reader = MockLandmarkReader(tree)
    elif args.backend == "replay":
        replay_kind = _replay_format(args.replay_file)
        reader = (
            CombinedReplayReader(args.replay_file, loop=not args.no_loop) if replay_kind == "combined"
            else ReplayLandmarkReader(args.replay_file, loop=not args.no_loop)
        )
    else:
        reader = (
            MocapBvhJointAngleReader(args.port)
            if args.mocap_drive == "bvh-joints"
            else MocapLandmarkReader(args.port)
        )
    if reader is not None:
        reader.connect()

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
    if args.hand_retargeter == "anydex" and args.hand_backend == "quest":
        config_paths = {"left": args.anydex_config_left, "right": args.anydex_config_right}
        anydex_hand_retargeters = {
            side: AnyDexDg5fRetargeter(side, config_paths[side]) for side in ("left", "right")
        }

    last_hand: Dict[str, object] = {"left": None, "right": None}
    calibrated = args.backend not in ("quest",)  # quest/direct-bvh backends have no calibrate() step to wait for
    tick_interval = 1.0 / args.hz
    physics_substeps = max(1, round(tick_interval / model.opt.timestep)) if args.sim_mode == "physics" else 0
    warned_missing = False
    last_reach_warning: Dict[str, float] = {"left": 0.0, "right": 0.0}

    def current_hand_joints() -> Dict[str, float]:
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
        return angles

    print(
        f"[upper_body_mujoco] backend={args.backend}, hand_backend={args.hand_backend}, "
        f"hand_retargeter={args.hand_retargeter}, mocap_drive={args.mocap_drive}, "
        f"poll={args.hz:.0f}Hz -- head frozen (no head DOF "
        "on the real robot), close the viewer window to stop"
    )

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Fixed viewpoint: no head DOF on the real robot means no reason for
        # the viewer to orbit/follow anything either -- set once, don't
        # re-home every frame (the user can still drag/zoom interactively;
        # this only sets the *starting* pose instead of mujoco.viewer's
        # default auto-framed one).
        viewer.cam.lookat[:] = [0.0, 0.0, 1.0]
        viewer.cam.distance = 2.5
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -15

        # Hide collision-only geom groups: the URDF importer puts <collision>
        # geoms (the coarse STL/capsule shapes physics actually uses) and
        # <visual> geoms (the detailed .obj meshes from mujoco_urdf.py) in
        # different groups, but mujoco.viewer's default geomgroup mask shows
        # BOTH -- overlapping the coarse collision shape on top of the
        # detailed one made the model look "simplified" even though nothing
        # was actually missing. A group is collision-only if every geom in
        # it has contype/conaffinity set (i.e. no group mixes visual and
        # collision geoms, true for this URDF); those are turned off here,
        # not deleted, so physics/contacts (once this stops being a pure
        # kinematic puppet) are unaffected.
        collision_only = [True] * len(viewer.opt.geomgroup)
        for i in range(model.ngeom):
            group = model.geom_group[i]
            if 0 <= group < len(collision_only):
                if model.geom_contype[i] == 0 and model.geom_conaffinity[i] == 0:
                    collision_only[group] = False
        for group, is_collision_only in enumerate(collision_only):
            if is_collision_only:
                viewer.opt.geomgroup[group] = 0

        last_tick = 0.0
        while viewer.is_running():
            now = time.monotonic()
            if now - last_tick < tick_interval:
                time.sleep(max(0.0, tick_interval - (now - last_tick)))
                continue
            last_tick = now

            if args.backend == "quest":
                left_hs, right_hs = hand_reader.poll()
                if left_hs is not None:
                    last_hand["left"] = left_hs
                if right_hs is not None:
                    last_hand["right"] = right_hs
                while quest_calibration_commands is not None:
                    try:
                        quest_calibration.request_capture(quest_calibration_commands.get_nowait())
                    except queue.Empty:
                        break
                if not quest_calibration.update(last_hand, now):
                    continue
                joints = None
                for side in ("left", "right"):
                    sample = last_hand.get(side)
                    if not _is_quest_hand(sample):
                        continue
                    wrist_pos, wrist_rotation = quest_calibration.wrist_target(
                        side, sample, args.wrist_scale
                    )
                    # Direction-preserving clamp onto the max-reach sphere
                    # (ported from refer/HapticSyncController.cs's MaxRadius
                    # clamp) instead of dropping the frame and holding the
                    # previous pose.
                    from_shoulder = wrist_pos - quest_shoulder[side]
                    reach = float(np.linalg.norm(from_shoulder))
                    if reach > quest_reach_limit[side]:
                        if now - last_reach_warning[side] > 1.0:
                            last_reach_warning[side] = now
                            print(
                                f"[upper_body_mujoco] {side} wrist target clamped to max reach "
                                f"({reach:.3f}m -> {quest_reach_limit[side]:.3f}m)"
                            )
                        wrist_pos = quest_shoulder[side] + from_shoulder * (quest_reach_limit[side] / reach)
                    joints = retargeter.solve_wrist_pose(side, wrist_pos, wrist_rotation)
                if joints is None:
                    continue
            else:
                if args.backend == "mocapapi" and args.mocap_drive == "bvh-joints":
                    joints = reader.poll_joint_angles(tree)
                    if joints is None:
                        continue
                else:
                    frame = reader.poll()
                    if frame is None:
                        continue
                    if landmark_panel is not None:
                        landmark_panel.update(frame)
                if hand_reader is not None:
                    left_hs, right_hs = hand_reader.poll()
                    if left_hs is not None:
                        last_hand["left"] = left_hs
                    if right_hs is not None:
                        last_hand["right"] = right_hs
                else:
                    embedded = getattr(reader, "last_hand", None)
                    if embedded is not None:
                        last_hand = embedded
                if not (args.backend == "mocapapi" and args.mocap_drive == "bvh-joints"):
                    if not calibrated:
                        retargeter.calibrate(frame)
                        calibrated = True
                        print(f"[upper_body_mujoco] calibrated ({args.calibration_pose})")
                    joints = retargeter.update(frame, retarget_torso=args.torso == "retarget")
                    if mink_ik is not None:
                        joints.update(mink_ik.update(frame))

            hand_joints = current_hand_joints()
            all_joints = {**joints, **hand_joints}
            missing = writer.write_ctrl(data, all_joints) if args.sim_mode == "physics" else writer.write(data, all_joints)
            if missing and not warned_missing:
                warned_missing = True
                print(
                    f"[upper_body_mujoco] WARNING: {missing} retargeted joint name(s) have no "
                    "matching MuJoCo joint this tick (name mismatch between rby1_dg5f.urdf and "
                    "the retargeter/hand retargeter's own joint names?) -- printed once"
                )

            if args.sim_mode == "physics":
                for _ in range(physics_substeps):
                    mujoco.mj_step(model, data)  # real gravity/contact dynamics -- ctrl is a servo target, not the pose
            else:
                mujoco.mj_forward(model, data)  # kinematics only, no mj_step -- see --sim-mode help
            collision_monitor.check(data, now)
            viewer.sync()

    if reader is not None:
        reader.close()
    if hand_reader is not None:
        hand_reader.close()
    if landmark_panel is not None:
        landmark_panel.close()


if __name__ == "__main__":
    main()
