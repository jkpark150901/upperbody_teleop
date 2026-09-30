"""Landmark-based Perception Neuron -> RBY1 upper-body retargeting."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import yaml

from robot_hand.urdf_fk import UrdfTree
from teleop.geometry import Pose

# Human BVH bone <-> robot link tracker map, same name-to-name pairing
# design as humanoid-retarget's tracker_dict / GMR's ik_match_table (see
# configs/upper_body/bvh_tracker.yaml and bvh_upper_body_mapping.md). This
# project's IK stays the closed-form analytic solve below, not a generic
# weighted-task QP solve -- only the *mapping table* is borrowed.
_TRACKER_CONFIG_PATH = pathlib.Path(__file__).resolve().parent.parent / "configs" / "upper_body" / "bvh_tracker.yaml"


def _load_bvh_tracker_config(path: pathlib.Path = _TRACKER_CONFIG_PATH):
    with open(path, "r", encoding="utf-8") as f:
        trackers = yaml.safe_load(f)["trackers"]
    robot_landmark_links = {key: entry["robot"] for key, entry in trackers.items()}
    mocap_names = {key: tuple(entry["human"]) for key, entry in trackers.items()}
    orientation_keys = {key for key, entry in trackers.items() if entry.get("orientation")}
    return robot_landmark_links, mocap_names, orientation_keys


_ROTATION_FIELDS = {"head_rotation", "left_wrist_rotation", "right_wrist_rotation", "bvh_joint_angles"}


@dataclass
class UpperBodyLandmarks:
    hips: np.ndarray
    chest: np.ndarray
    head: np.ndarray
    left_shoulder: np.ndarray
    left_elbow: np.ndarray
    left_wrist: np.ndarray
    right_shoulder: np.ndarray
    right_elbow: np.ndarray
    right_wrist: np.ndarray
    head_rotation: np.ndarray | None = None
    # 3x3 world rotation matrices for the hand markers, same shape/meaning as
    # head_rotation. Optional -- readers that only capture positions (replay
    # recordings) leave these None and the wrist-orientation IK block below
    # is simply skipped, exactly like the existing head_rotation fallback.
    left_wrist_rotation: np.ndarray | None = None
    right_wrist_rotation: np.ndarray | None = None
    # Optional approximate robot-joint angle hints derived from BVH local
    # rotations. These are not written directly to the robot; they bias
    # IK branch selection so the solved pose keeps the BVH joint direction
    # where multiple robot configurations reach the same landmarks.
    bvh_joint_angles: Dict[str, float] | None = None

    def as_dict(self) -> Dict[str, np.ndarray]:
        return {
            name: np.asarray(value, dtype=float)
            for name, value in vars(self).items()
            if name not in _ROTATION_FIELDS
        }


JOINT_NAMES = (
    "torso_hp", "torso_5",
    *(f"left_arm_{i}" for i in range(7)),
    *(f"right_arm_{i}" for i in range(7)),
    "head_0", "head_1",
)

ROBOT_LANDMARK_LINKS, MOCAP_TRACKER_NAMES, ORIENTATION_TRACKER_KEYS = _load_bvh_tracker_config()

# DG5F palm's orientation relative to link_{side}_arm_6: 180 deg about X,
# from the left_hand_mount / right_hand_mount fixed joints in the URDF
# (identical on both sides -- the dg_base/palm joints below the mount add
# no further rotation, only translation). Self-inverse (a 180 deg rotation
# undoes itself), so it's used both palm->arm6 and arm6->palm below.
_HAND_MOUNT_ROTATION = np.array([
    [1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, 0.0, -1.0],
])

# DG5F palm's own local axes (left_hand_ll_dg_palm / right_hand_rl_dg_palm,
# same convention _quest_forward_reference() in upper_body_retarget_mujoco.py
# uses): local +Z points toward the fingers, local +X is the back-of-hand
# normal (dorsal) -- encoded directly as column order in
# _wrist_rotation_target()'s palm_target construction below.

# BVH/MocapApi's LeftHand/RightHand bone local axes, confirmed on real
# hardware: dorsal (back of hand) faces local +Y on *both* sides, but
# fingers flip sign between sides -- RightHand's local -X points along the
# fingers same as the initial (right-side) check found, but LeftHand's
# rigging mirrors that component, so LeftHand's fingers direction is local
# +X instead. (Same kind of left/right local-axis mirroring AnyDexRetarget
# already has to special-case for other rigs -- see vr_device_dg5f_
# retargeting.md's KeyVectorOptimizer section.)
_BVH_HAND_FINGERS_LOCAL = {"left": np.array([1.0, 0.0, 0.0]), "right": np.array([-1.0, 0.0, 0.0])}
_BVH_HAND_DORSAL_LOCAL = np.array([0.0, -1.0, 0.0])


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else np.zeros(3)


def _kabsch(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rotation mapping centered source vectors onto target vectors."""
    h = source.T @ target
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1] *= -1
        r = vt.T @ u.T
    return r


class RBY1UpperBodyRetargeter:
    """Scale-free landmark retargeting followed by bounded damped-least-squares IK."""

    def __init__(self, tree: UrdfTree, root_pose: Pose | None = None, calibration_pose: str = "attention"):
        self.tree = tree
        self.root_pose = root_pose or Pose.identity()
        self.names = list(JOINT_NAMES)
        self.lower = np.array([tree.joints[n].lower for n in self.names])
        self.upper = np.array([tree.joints[n].upper for n in self.names])
        self.q = np.zeros(len(self.names))
        self.reference_q = np.zeros(len(self.names))
        self.calibration_pose = "attention"
        self._human_to_robot: np.ndarray | None = None
        self._neutral_head_rotation: np.ndarray | None = None
        self._neutral_bvh_joint_angles: Dict[str, float] | None = None
        # Last valid (non-degenerate) shoulder-elbow-wrist plane normal per
        # side -- see _solve_arm_position_block's straight-arm fallback.
        self._prev_plane_n: Dict[str, np.ndarray | None] = {"left": None, "right": None}
        self._neutral_robot = self._robot_points(self.q)
        self.set_calibration_pose(calibration_pose)
        # Diagnostics from the most recent update(): where each landmark's
        # IK target was, and how far the converged robot pose ended up from
        # it. A small error means the IK is doing its job and any "wrong
        # motion" is coming from the target itself (calibration/_targets()
        # math); a large, persistent error means the IK isn't reaching a
        # target it's been given (joint limits, too few iterations, or a
        # genuinely unreachable target) -- two different bugs to chase.
        self.last_target: Dict[str, np.ndarray] = {}
        self.last_error_m: Dict[str, float] = {}
        # Same idea, for _solve_wrist_orientation_block: degrees between the
        # requested wrist orientation and what the clipped (4,5,6) angles
        # actually reach. Near 0 means the ZYZ solve is exact and any
        # "wrong-looking" hand orientation is coming from the target itself
        # (_wrist_rotation_target()); a large, persistent value means the
        # target is outside arm_5's asymmetric [-90,+110 deg] (or another
        # joint's) reachable range and clipping is distorting the result --
        # two different bugs to chase, same as last_error_m above.
        self.last_wrist_orientation_error_deg: Dict[str, float] = {}
        # Off by default -- see _extension_ratio()/_measure_human_lengths().
        # When on, each segment's robot-length is scaled by how extended the
        # *human's own* BVH segment currently is relative to its length at
        # calibration time, instead of always snapping to the robot's full
        # segment length regardless of how bent/relaxed the human's arm is.
        self.use_bvh_scale = False
        self._human_reference_lengths: Dict[str, float] = {}

    def _reference_q(self, pose: str) -> np.ndarray:
        """Joint vector for a named calibration pose, clipped to limits."""
        if pose not in ("attention", "tpose", "raise"):
            raise ValueError("calibration pose must be 'attention', 'tpose', or 'raise'")
        q = np.zeros(len(self.names))
        if pose == "tpose":
            # At zero the RBY1 arms point down. Shoulder X rotation lifts them
            # sideways: +90 deg left, -90 deg right (mirrored -- sideways
            # extension is a mirror-image motion between the two arms).
            q[self.names.index("left_arm_1")] = np.pi / 2.0
            q[self.names.index("right_arm_1")] = -np.pi / 2.0
        elif pose == "raise":
            # Both arms straight up overhead ("만세") -- continuing tpose's
            # same lateral-lift axis (arm_1) the rest of the way to its
            # limit (left: +pi, right: -pi -- both exactly in range) swings
            # the arm from "down" through "sideways" to "up", landing the
            # wrist directly above its own shoulder (checked against
            # forward_kinematics, not guessed -- left wrist ends up at
            # (0, 0.22, 0.61) relative to chest, y matching the shoulder's
            # own offset and z clearly the highest of any reference pose,
            # vs. e.g. driving arm_0 instead which lands lower and crossed
            # toward the body's centerline, a less natural "raise" shape).
            # An earlier "reach forward" pose here used an unverified
            # guessed angle and was found to break real tracking -- see
            # git history; this replaced it after checking the above.
            q[self.names.index("left_arm_1")] = np.pi
            q[self.names.index("right_arm_1")] = -np.pi
        return np.clip(q, self.lower, self.upper)

    def set_calibration_pose(self, pose: str) -> None:
        """Select the robot pose that the next human calibration frame represents."""
        self.reference_q = self._reference_q(pose)
        self.q = self.reference_q.copy()
        self.calibration_pose = pose
        self._neutral_robot = self._robot_points(self.reference_q)
        self._human_to_robot = None
        self._neutral_head_rotation = None
        self._neutral_bvh_joint_angles = None
        self._prev_plane_n = {"left": None, "right": None}

    def _values(self, q: np.ndarray) -> Dict[str, float]:
        return dict(zip(self.names, q))

    def poses(self, q: np.ndarray | None = None):
        return self.tree.forward_kinematics(self._values(self.q if q is None else q), self.root_pose)

    def _robot_points(self, q: np.ndarray) -> Dict[str, np.ndarray]:
        # This runs ~(1 + active joints) times per IK iteration for the
        # numerical Jacobian -- full-tree FK would also walk both hands'
        # ~30 finger joints each time for no reason, so restrict it to the
        # ancestor chains of the 8 landmark links actually used below.
        poses = self.tree.forward_kinematics(
            self._values(q), self.root_pose, only=ROBOT_LANDMARK_LINKS.values()
        )
        return {name: poses[link].pos.copy() for name, link in ROBOT_LANDMARK_LINKS.items()}

    def _measure_human_lengths(self, h: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Per-segment human bone lengths from a BVH-derived landmark frame
        (same shape as _targets()' own per-segment split) -- the reference
        --bvh-scale measures every later frame's segments against, so a
        95%-extended elbow gives ~95% of the robot's own segment length
        instead of _targets()' default of always using 100% regardless of
        how bent/relaxed the human's arm actually is.
        """
        lengths = {}
        for side in ("left", "right"):
            shoulder_key, elbow_key, wrist_key = f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist"
            lengths[shoulder_key] = float(np.linalg.norm(h[shoulder_key] - h["chest"]))
            lengths[f"{side}_upper"] = float(np.linalg.norm(h[elbow_key] - h[shoulder_key]))
            lengths[f"{side}_lower"] = float(np.linalg.norm(h[wrist_key] - h[elbow_key]))
        lengths["head"] = float(np.linalg.norm(h["head"] - h["chest"]))
        return lengths

    def _extension_ratio(self, key: str, current_len: float) -> float:
        """current_len / (that segment's length at calibration time), clamped
        to [0, 1] -- 1.0 (no scaling down) if --bvh-scale is off, there's no
        reference yet, or the reference was degenerately short."""
        if not self.use_bvh_scale:
            return 1.0
        ref = self._human_reference_lengths.get(key)
        if not ref or ref < 1e-6:
            return 1.0
        return float(np.clip(current_len / ref, 0.0, 1.0))

    def calibrate(self, human: UpperBodyLandmarks) -> None:
        h = human.as_dict()
        hp = np.stack([
            h["head"] - h["chest"],
            h["left_shoulder"] - h["chest"],
            h["right_shoulder"] - h["chest"],
        ])
        r = self._neutral_robot
        rp = np.stack([
            r["head"] - r["chest"],
            r["left_shoulder"] - r["chest"],
            r["right_shoulder"] - r["chest"],
        ])
        self._human_to_robot = _kabsch(hp, rp)
        self._neutral_head_rotation = (
            None if human.head_rotation is None else np.asarray(human.head_rotation).copy()
        )
        self._neutral_bvh_joint_angles = None if human.bvh_joint_angles is None else dict(human.bvh_joint_angles)
        self._human_reference_lengths = self._measure_human_lengths(h)
        self.q = self.reference_q.copy()

    def calibrate_multi(self, frames: Dict[str, UpperBodyLandmarks], active_pose: str = "attention") -> None:
        """Fit the calibration rotation jointly from several reference poses
        (e.g. tpose + attention + raise) instead of just one -- pools each
        pose's 3 vectors (chest->head, chest->left_shoulder,
        chest->right_shoulder) against that *same* pose's own robot
        reference into one bigger Kabsch fit, so calibration isn't riding
        on a single snapshot. `frames` keys must be calibration pose names
        ("attention"/"tpose"/"raise"); `active_pose` selects which one's
        rest configuration stays in effect for _targets()/update()
        afterward (independent of how many frames were pooled into the
        rotation fit).

        Segment lengths (_human_reference_lengths, what --bvh-scale scales
        against) are taken as the *max* measured across every captured
        pose, not just active_pose's -- a bent elbow/shoulder always
        measures short, so whichever pose happens to extend a given
        segment most fully gives the best estimate of its true length; a
        single frame can easily underestimate every segment it didn't
        happen to fully extend.
        """
        human_vecs, robot_vecs = [], []
        for pose_name, human in frames.items():
            h = human.as_dict()
            r = self._robot_points(self._reference_q(pose_name))
            human_vecs += [h["head"] - h["chest"], h["left_shoulder"] - h["chest"], h["right_shoulder"] - h["chest"]]
            robot_vecs += [r["head"] - r["chest"], r["left_shoulder"] - r["chest"], r["right_shoulder"] - r["chest"]]
        rotation = _kabsch(np.stack(human_vecs), np.stack(robot_vecs))

        self.set_calibration_pose(active_pose)  # resets reference_q/_neutral_robot/q; clears _human_to_robot
        self._human_to_robot = rotation
        active_frame = frames[active_pose]
        self._neutral_head_rotation = (
            None if active_frame.head_rotation is None else np.asarray(active_frame.head_rotation).copy()
        )
        self._neutral_bvh_joint_angles = (
            None if active_frame.bvh_joint_angles is None else dict(active_frame.bvh_joint_angles)
        )
        per_pose_lengths = [self._measure_human_lengths(human.as_dict()) for human in frames.values()]
        self._human_reference_lengths = {
            key: max(lengths[key] for lengths in per_pose_lengths) for key in per_pose_lengths[0]
        }

    def _targets(self, human: UpperBodyLandmarks) -> Dict[str, np.ndarray]:
        if self._human_to_robot is None:
            self.calibrate(human)
        a = self._human_to_robot
        h = human.as_dict()
        r = self._neutral_robot
        chest = r["chest"]
        target = {"chest": chest}

        for side in ("left", "right"):
            shoulder_key, elbow_key, wrist_key = (
                f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist"
            )
            shoulder_len = np.linalg.norm(r[shoulder_key] - chest)
            upper_len = np.linalg.norm(r[elbow_key] - r[shoulder_key])
            lower_len = np.linalg.norm(r[wrist_key] - r[elbow_key])
            if self.use_bvh_scale:
                shoulder_len *= self._extension_ratio(
                    shoulder_key, np.linalg.norm(h[shoulder_key] - h["chest"])
                )
                upper_len *= self._extension_ratio(
                    f"{side}_upper", np.linalg.norm(h[elbow_key] - h[shoulder_key])
                )
                lower_len *= self._extension_ratio(
                    f"{side}_lower", np.linalg.norm(h[wrist_key] - h[elbow_key])
                )
            shoulder = chest + shoulder_len * _unit(a @ (h[shoulder_key] - h["chest"]))
            elbow = shoulder + upper_len * _unit(a @ (h[elbow_key] - h[shoulder_key]))
            wrist = elbow + lower_len * _unit(a @ (h[wrist_key] - h[elbow_key]))
            target[shoulder_key], target[elbow_key], target[wrist_key] = shoulder, elbow, wrist

        head_len = np.linalg.norm(r["head"] - chest)
        if self.use_bvh_scale:
            head_len *= self._extension_ratio("head", np.linalg.norm(h["head"] - h["chest"]))
        target["head"] = chest + head_len * _unit(a @ (h["head"] - h["chest"]))
        return target

    def _chain_points(self, q: np.ndarray, keys: Tuple[str, ...]) -> Dict[str, np.ndarray]:
        links = [ROBOT_LANDMARK_LINKS[k] for k in keys]
        poses = self.tree.forward_kinematics(self._values(q), self.root_pose, only=links)
        return {k: poses[ROBOT_LANDMARK_LINKS[k]].pos.copy() for k in keys}

    def _solve_block(
        self,
        active_names: Tuple[str, ...],
        keys: Tuple[str, ...],
        target: Dict[str, np.ndarray],
        q_start: np.ndarray,
        iterations: int,
        points_fn=None,
    ) -> None:
        """Damped least-squares IK restricted to one set of joints/targets.

        `active_names`'s joints only ever move `keys`' landmark links (true
        for a single arm, or the head, once the torso is pinned -- they
        share no kinematic chain), so solving each block on its own gives
        the exact same result as one big joint solve over everything, just
        without wasting Jacobian columns and forward_kinematics calls on
        links that block can't actually move.

        `points_fn(q) -> Dict[key, np.ndarray]` overrides how `keys` are
        turned into world points -- default is self._chain_points (real
        landmark links). The 3-1-3 shoulder/wrist-orientation blocks below
        pass a points_fn that reads *virtual* marker points (a fixed local
        offset applied to a real link's pose) so a position-only DLS solve
        can also pin down a swivel angle or an orientation, reusing this
        same solver/active-set-joint-limit machinery unchanged.
        """
        active = np.array([self.names.index(n) for n in active_names])
        if points_fn is None:
            points_fn = lambda q: self._chain_points(q, keys)  # noqa: E731

        def residual(q: np.ndarray) -> np.ndarray:
            points = points_fn(q)
            position_error = np.concatenate([target[k] - points[k] for k in keys])
            # Keeps redundant wrist/twist axes smooth while position IK is underconstrained.
            return np.concatenate([position_error, 0.025 * (q_start[active] - q[active])])

        for _ in range(iterations):
            e = residual(self.q)
            j = np.empty((e.size, active.size))
            eps = 1e-4
            for column, i in enumerate(active):
                q2 = self.q.copy()
                q2[i] += eps
                j[:, column] = (residual(q2) - e) / eps
            # residual is target-current, hence the minus sign.
            dq = -j.T @ np.linalg.solve(j @ j.T + 2e-3 * np.eye(e.size), e)

            # Active-set joint limits. A joint sitting exactly at a limit
            # (e.g. the elbow at its straight-arm bound in the reference
            # pose) still gets a normal, often large, Jacobian column if the
            # *unconstrained* direction would reduce the residual -- but
            # np.clip below silently throws that motion away once dq is
            # applied. Left in the shared least-squares solve, that column
            # "claims" residual reduction it can never deliver, so the other
            # genuinely free joints get under-corrected and convergence
            # stalls or oscillates (a wrist target 12cm from a fully
            # reachable neutral pose was measured settling ~970mm off
            # before this fix). Detect those columns and re-solve with only
            # the free ones so their correction isn't stolen.
            at_upper = self.q[active] >= self.upper[active] - 1e-9
            at_lower = self.q[active] <= self.lower[active] + 1e-9
            blocked = (at_upper & (dq > 0)) | (at_lower & (dq < 0))
            if blocked.any():
                free = ~blocked
                if free.any():
                    jf = j[:, free]
                    dqf = -jf.T @ np.linalg.solve(jf @ jf.T + 2e-3 * np.eye(e.size), e)
                    dq = np.zeros_like(dq)
                    dq[free] = dqf
                else:
                    dq = np.zeros_like(dq)

            self.q[active] = np.clip(
                self.q[active] + np.clip(dq, -0.12, 0.12),
                self.lower[active],
                self.upper[active],
            )
            if np.linalg.norm(e[: 3 * len(keys)]) < 2e-3:
                break

    def _shoulder_theta0(self, side: str) -> float:
        """Fixed change-of-basis angle for this side's shoulder -- see
        _solve_arm_position_block. arm_0's own axis is tilted (not a canonical
        X/Y/Z axis) but always perpendicular to arm_1's axis (world +X) in
        this URDF; theta0 is the rotation about X that carries canonical Z
        onto arm_0's axis, i.e. Rx(theta0) @ (0,0,1) == arm_0.axis.
        """
        axis0 = self.tree.joints[f"{side}_arm_0"].axis
        return float(np.arctan2(-axis0[1], axis0[2]))

    @staticmethod
    def _zxz_candidates(r_target: np.ndarray, theta0: float) -> list:
        """Both (q0, q1, q2) solutions of
        axis_angle(axis0, q0) @ Rx(q1) @ Rz(q2) == r_target, exploiting
        that this is a textbook proper-Euler ZXZ decomposition once
        rotated into the axis0-aligned frame (arm_0's axis -> canonical Z,
        arm_1's axis (+X) is already canonical and unaffected by that
        change of basis, since it's the rotation axis of that same
        change). Verified by reconstructing r_target from each candidate
        via the actual joint axes and checking against forward_kinematics
        directly -- see the dev notes this was built with. Falls back to
        the alpha+gamma-only-determined gimbal-lock case when sin(q1 -
        theta0) is ~0 (q1 - theta0 ~ 0 or pi).
        """
        c0, s0 = np.cos(theta0), np.sin(theta0)
        c = np.array([[1.0, 0.0, 0.0], [0.0, c0, -s0], [0.0, s0, c0]])
        m = c.T @ r_target
        beta1 = np.arccos(np.clip(m[2, 2], -1.0, 1.0))
        candidates = []
        for beta in (beta1, -beta1):
            s = np.sin(beta)
            if abs(s) > 1e-6:
                gamma = np.arctan2(m[2, 0] / s, m[2, 1] / s)
                alpha = np.arctan2(m[0, 2] / s, -m[1, 2] / s)
            else:
                gamma = 0.0
                alpha = np.arctan2(-m[0, 1], m[0, 0])
            candidates.append((alpha, beta + theta0, gamma))
        return candidates

    @staticmethod
    def _wrap_pi(angle: float) -> float:
        return float((angle + np.pi) % (2 * np.pi) - np.pi)

    def _bvh_joint_prior(self, human: UpperBodyLandmarks | None) -> Dict[str, float] | None:
        """Approximate robot joint targets from BVH local-angle deltas.

        Used only as a branch/smoothness prior. Landmark positions still
        define the IK target, so a noisy or structurally-mismatched BVH
        Euler channel cannot pull the wrist away from the captured pose.
        """
        if (
            human is None
            or human.bvh_joint_angles is None
            or self._neutral_bvh_joint_angles is None
        ):
            return None
        prior = {}
        for name, value in human.bvh_joint_angles.items():
            if name not in self.names:
                continue
            neutral = self._neutral_bvh_joint_angles.get(name)
            if neutral is None:
                continue
            i = self.names.index(name)
            target = self.reference_q[i] + self._wrap_pi(float(value) - float(neutral))
            prior[name] = float(np.clip(target, self.lower[i], self.upper[i]))
        return prior

    def _solve_arm_position_block(
        self,
        side: str,
        target: Dict[str, np.ndarray],
        q_start: np.ndarray,
        joint_prior: Dict[str, float] | None = None,
    ) -> None:
        """Shoulder (arm_0/1/2, spherical) + elbow (arm_3, single-DOF)
        solved together, in closed form. Covers the "3" and the "1" of the
        arm's 3-1-3 structure in one pass -- elbow position only depends
        on the shoulder's aim, and wrist position only depends on
        shoulder+elbow together, so they can't be solved independently.
        The wrist's own "3" (orientation) never moves position at all and
        is handled separately, by _solve_wrist_orientation_block.

        The shoulder's target rotation is built from two orthonormal
        frames -- body-frame {upper-arm axis, arm_2's local +Y} and
        world-frame {aim direction, a swivel-plane normal} -- decomposed
        into (q0,q1,q2) via _zxz_candidates instead of an iterative DLS
        search (an earlier DLS version, matching two virtual marker points
        by Jacobian descent, got stuck in local minima on real targets --
        see git history). The swivel plane is the human's actual
        shoulder-elbow-wrist plane, but its *sign* is ambiguous (the
        normal only pins down the plane, not which of its two directions
        is arm_2's actual +Y) -- crossed with _zxz_candidates' own
        two-solution ambiguity, that's up to 4 shoulder branches. Each is
        paired with its own exact elbow angle: whichever swivel branch was
        picked, local +Y is by the URDF's own geometry always
        perpendicular to both bone segments, so *some* elbow angle reaches
        the wrist target exactly -- solved for directly (another atan2, no
        search) rather than guessed at. The branch actually used is
        whichever needs the least joint-limit clipping to reach the wrist,
        tie-broken by smoothness (closest to q_start).
        """
        shoulder_key, elbow_key, wrist_key = f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist"
        aim_dir = _unit(target[elbow_key] - target[shoulder_key])
        forearm_dir = _unit(target[wrist_key] - target[elbow_key])
        plane_n = _unit(np.cross(target[elbow_key] - target[shoulder_key], target[wrist_key] - target[elbow_key]))
        if np.linalg.norm(plane_n) < 1e-6:
            # Degenerate (perfectly straight arm): shoulder/elbow/wrist are
            # ~colinear, so the plane -- and hence the swivel -- is
            # undefined. Hold the last valid one instead of guessing.
            plane_n = self._prev_plane_n.get(side)
            if plane_n is None:
                fallback_axis = np.array([0.0, 0.0, 1.0])
                cross = np.cross(aim_dir, fallback_axis)
                if np.linalg.norm(cross) < 1e-6:
                    fallback_axis = np.array([1.0, 0.0, 0.0])
                    cross = np.cross(aim_dir, fallback_axis)
                plane_n = _unit(cross)
        else:
            self._prev_plane_n[side] = plane_n

        # Fixed local offsets read straight from the URDF: u0 is the
        # elbow joint's own <origin> (shoulder -> elbow, in the shoulder's
        # unrotated frame -- what arm_0/1/2 actually rotate), v2 is
        # arm_4's <origin> (elbow -> wrist, in the elbow's unrotated frame
        # -- what arm_3 actually rotates).
        u0_hat = _unit(self.tree.joints[f"{side}_arm_3"].origin.pos)
        v2_hat = _unit(self.tree.joints[f"{side}_arm_4"].origin.pos)
        ang_v2 = float(np.arctan2(v2_hat[0], v2_hat[2]))
        theta0 = self._shoulder_theta0(side)
        body = np.stack([u0_hat, np.array([0.0, 1.0, 0.0]), np.cross(u0_hat, [0.0, 1.0, 0.0])], axis=1)

        sh_idx = np.array([self.names.index(f"{side}_arm_{i}") for i in range(3)])
        i3 = self.names.index(f"{side}_arm_3")
        idx4 = np.concatenate([sh_idx, [i3]])
        pivot_link, wrist_link = f"link_{side}_arm_2", f"link_{side}_arm_6"

        best = None
        for sign in (1.0, -1.0):
            n = sign * plane_n
            b = np.stack([aim_dir, n, np.cross(aim_dir, n)], axis=1)
            r_target = b @ body.T
            for q0, q1, q2 in self._zxz_candidates(r_target, theta0):
                qtest = self.q.copy()
                qtest[sh_idx] = (q0, q1, q2)
                pivot_r = self.tree.forward_kinematics(
                    self._values(qtest), self.root_pose, only=[pivot_link]
                )[pivot_link].as_matrix()
                req_local = pivot_r.T @ forearm_dir
                q3 = self._wrap_pi(float(np.arctan2(req_local[0], req_local[2])) - ang_v2)

                full = np.array([q0, q1, q2, q3])
                violation = float(np.sum(
                    np.clip(self.lower[idx4] - full, 0, None) + np.clip(full - self.upper[idx4], 0, None)
                ))
                clipped = np.clip(full, self.lower[idx4], self.upper[idx4])
                qtest[idx4] = clipped
                wpos = self.tree.forward_kinematics(
                    self._values(qtest), self.root_pose, only=[wrist_link]
                )[wrist_link].pos
                wrist_err = float(np.linalg.norm(wpos - target[wrist_key]))
                if joint_prior:
                    prior_vec = np.array([
                        joint_prior.get(f"{side}_arm_{i}", q_start[self.names.index(f"{side}_arm_{i}")])
                        for i in range(4)
                    ])
                    prior_err = float(np.linalg.norm(clipped - prior_vec))
                else:
                    prior_err = 0.0
                score = (
                    round(violation, 6),
                    round(wrist_err, 6),
                    round(prior_err, 6),
                    float(np.linalg.norm(clipped - q_start[idx4])),
                )
                if best is None or score < best[0]:
                    best = (score, clipped)

        self.q[idx4] = best[1]

    def _wrist_rotation_target(self, side: str, human: UpperBodyLandmarks) -> np.ndarray | None:
        """World rotation the robot's link_{side}_arm_6 should reach, or
        None if this frame has no wrist rotation to go on (same
        None-means-skip convention as head_rotation).

        Built directly from BVH's own known hand-local axis semantics and
        the DG5F palm's own known local axis semantics (both confirmed on
        real hardware -- see the module-level _BVH_HAND_*/_PALM_* comments),
        composed through the URDF's fixed dg_base/palm mount offset
        (_HAND_MOUNT_ROTATION, 180 deg about X from link_{side}_arm_6).

        This replaced an earlier neutral-delta approach (conjugating a
        calibration-relative wrist rotation delta by the Kabsch
        body-alignment rotation _human_to_robot) that solved wrong relative
        to dg_base: that conjugation is only valid if the human wrist's own
        local rest frame happens to line up with _human_to_robot, which was
        fit from torso/shoulder/head *positions* and has no established
        relationship to the wrist's local axis convention. Tracking the
        fingers/dorsal *directions* explicitly sidesteps that -- direction
        vectors transform correctly under _human_to_robot regardless of
        either frame's local axis convention, since it's fit exactly for
        that (position-vector-direction) purpose.
        """
        current = human.left_wrist_rotation if side == "left" else human.right_wrist_rotation
        if current is None:
            return None
        current = np.asarray(current, dtype=float)
        a = self._human_to_robot

        fingers_world = _unit(a @ (current @ _BVH_HAND_FINGERS_LOCAL[side]))
        dorsal_world = a @ (current @ _BVH_HAND_DORSAL_LOCAL)
        dorsal_world = _unit(dorsal_world - np.dot(dorsal_world, fingers_world) * fingers_world)
        third_world = np.cross(fingers_world, dorsal_world)  # local +Y, so (dorsal, +Y, fingers) stays right-handed

        palm_target = np.empty((3, 3))
        palm_target[:, 0] = dorsal_world
        palm_target[:, 1] = third_world
        palm_target[:, 2] = fingers_world
        return palm_target @ _HAND_MOUNT_ROTATION

    @staticmethod
    def _zyz_candidates(m: np.ndarray) -> list:
        """Both (q4, q5, q6) solutions of Rz(q4) @ Ry(q5) @ Rz(q6) == m --
        arm_4/5/6's axes (Z, Y, Z) are already canonical, a textbook proper
        Euler ZYZ decomposition with no change of basis needed (unlike the
        shoulder's tilted arm_0 axis -- see _zxz_candidates). Verified the
        same way: reconstructing m from each candidate and checking against
        forward_kinematics directly.
        """
        beta1 = np.arccos(np.clip(m[2, 2], -1.0, 1.0))
        candidates = []
        for beta in (beta1, -beta1):
            s = np.sin(beta)
            if abs(s) > 1e-6:
                gamma = np.arctan2(m[2, 1] / s, -m[2, 0] / s)
                alpha = np.arctan2(m[1, 2] / s, m[0, 2] / s)
            else:
                gamma = 0.0
                alpha = np.arctan2(-m[0, 1], m[0, 0])
            candidates.append((alpha, beta, gamma))
        return candidates

    def _solve_wrist_orientation_block(
        self,
        side: str,
        rotation_target: np.ndarray,
        q_start: np.ndarray,
        joint_prior: Dict[str, float] | None = None,
    ) -> None:
        """3-DOF wrist (arm_4/5/6) only, solved in closed form -- these
        three joints also share a single point (== the wrist landmark
        itself), so they never move wrist *position* at all, only
        orientation, which _zyz_candidates decomposes directly (see
        _solve_arm_position_block for why closed form replaced an earlier
        DLS version here too: it plateaus at a wrong, but joint-limit-
        stable-looking, orientation instead of finding the reachable
        branch). Picks whichever of the two ZYZ solutions needs the least
        joint-limit clipping, tie-broken by smoothness (closest to q_start).
        """
        link3 = f"link_{side}_arm_3"  # arm_4/5/6's parent frame (post shoulder+elbow)
        parent_r = self.tree.forward_kinematics(self._values(self.q), self.root_pose, only=[link3])[link3].as_matrix()
        m = parent_r.T @ rotation_target

        idx = np.array([self.names.index(f"{side}_arm_{i}") for i in (4, 5, 6)])

        def violation(c) -> float:
            v = np.array(c)
            return float(np.sum(np.clip(self.lower[idx] - v, 0, None) + np.clip(v - self.upper[idx], 0, None)))

        if joint_prior:
            prior_vec = np.array([
                joint_prior.get(f"{side}_arm_{i}", q_start[self.names.index(f"{side}_arm_{i}")])
                for i in (4, 5, 6)
            ])
        else:
            prior_vec = None

        best = min(
            self._zyz_candidates(m),
            key=lambda c: (
                round(violation(c), 6),
                round(0.0 if prior_vec is None else float(np.linalg.norm(np.array(c) - prior_vec)), 6),
                np.linalg.norm(np.array(c) - q_start[idx]),
            ),
        )
        self.q[idx] = np.clip(best, self.lower[idx], self.upper[idx])

        wrist_link = f"link_{side}_arm_6"
        achieved = self.tree.forward_kinematics(
            self._values(self.q), self.root_pose, only=[wrist_link]
        )[wrist_link].as_matrix()
        cos_angle = np.clip((np.trace(rotation_target.T @ achieved) - 1.0) / 2.0, -1.0, 1.0)
        self.last_wrist_orientation_error_deg[side] = float(np.degrees(np.arccos(cos_angle)))

    def solve_wrist(self, side: str, wrist_pos: np.ndarray, iterations: int = 7) -> Dict[str, float]:
        """Torso-fixed, one-arm, wrist-position-only IK.

        Skips calibrate()/_targets() (the scale-free segment-direction
        mapping + Kabsch alignment) entirely: `wrist_pos` is a target
        already expressed in the robot's own frame, and only that arm's 7
        joints move to reach it. Meant as a minimal test entry point to
        check "can/how-fast can the IK reach this point" in isolation from
        the rest of the retargeting pipeline -- see scripts/wrist_ik_test_vedo.py.
        """
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        self.q[:2] = self.reference_q[:2]
        q_start = self.q.copy()
        key = f"{side}_wrist"
        target = {key: np.asarray(wrist_pos, dtype=float)}
        self._solve_block(tuple(f"{side}_arm_{i}" for i in range(7)), (key,), target, q_start, iterations)
        self.last_target = target
        final_points = self._chain_points(self.q, (key,))
        self.last_error_m = {key: float(np.linalg.norm(target[key] - final_points[key]))}
        return self._values(self.q)

    def solve_wrist_pose(
        self,
        side: str,
        wrist_pos: np.ndarray,
        wrist_rotation: np.ndarray | None = None,
        iterations: int = 7,
    ) -> Dict[str, float]:
        """Torso-fixed one-arm IK for a wrist position plus optional rotation.

        This is the Quest-only counterpart to update()'s mocap path: Quest
        supplies a wrist pose directly, already in the robot frame after the
        caller's calibration/remap, so this skips landmark calibration and
        only drives that arm. Position is solved first; wrist orientation
        then uses the same closed-form arm_4/5/6 block as the full upper-body
        retargeter.
        """
        joints = self.solve_wrist(side, wrist_pos, iterations)
        if wrist_rotation is not None:
            q_start = self.q.copy()
            self._solve_wrist_orientation_block(side, np.asarray(wrist_rotation, dtype=float), q_start)
            joints = self._values(self.q)
        return joints

    def update(
        self,
        human: UpperBodyLandmarks,
        iterations: int = 7,
        retarget_torso: bool = False,
    ) -> Dict[str, float]:
        target = self._targets(human)
        q_start = self.q.copy()
        joint_prior = self._bvh_joint_prior(human)

        if retarget_torso:
            # Torso motion changes both arms' and the head's chain, so this
            # case keeps the single combined solve over everything.
            self._solve_block(tuple(self.names), tuple(ROBOT_LANDMARK_LINKS), target, q_start, iterations)
        else:
            # Torso pinned -> left arm, right arm, and head are independent
            # kinematic chains, and each arm is itself a 3-1-3 chain: a
            # spherical shoulder (arm_0/1/2), a single-DOF elbow (arm_3),
            # and a spherical wrist (arm_4/5/6) -- each group shares one
            # point and moves nothing the others' targets depend on (torso
            # pinned), so solving them as three small blocks per arm is
            # exact, not an approximation, same as the old shoulder/elbow/
            # wrist-as-one-block split was for whole arms vs. torso+head.
            self.q[:2] = self.reference_q[:2]
            for side in ("left", "right"):
                self._solve_arm_position_block(side, target, q_start, joint_prior)
                rotation_target = self._wrist_rotation_target(side, human)
                if rotation_target is not None:
                    self._solve_wrist_orientation_block(side, rotation_target, q_start, joint_prior)
            self._solve_block(("head_0", "head_1"), ("head",), target, q_start, iterations)

        if human.head_rotation is not None and self._neutral_head_rotation is not None:
            delta = self._neutral_head_rotation.T @ np.asarray(human.head_rotation)
            # Small-angle vector. Axis Neuron is Y-up: Y is head yaw and X is pitch;
            # RBY1 uses Z yaw (head_0) and Y pitch (head_1).
            rotvec = 0.5 * np.array([
                delta[2, 1] - delta[1, 2],
                delta[0, 2] - delta[2, 0],
                delta[1, 0] - delta[0, 1],
            ])
            yaw_i, pitch_i = self.names.index("head_0"), self.names.index("head_1")
            self.q[yaw_i] = np.clip(self.reference_q[yaw_i] + rotvec[1], self.lower[yaw_i], self.upper[yaw_i])
            self.q[pitch_i] = np.clip(self.reference_q[pitch_i] + rotvec[0], self.lower[pitch_i], self.upper[pitch_i])

        # Diagnostics: how far each landmark ended up from the target it was
        # given, using the truly final q (after the head-rotation override
        # above too). See the module-level comment on last_target/last_error_m.
        self.last_target = target
        final_points = self._robot_points(self.q)
        self.last_error_m = {
            key: float(np.linalg.norm(target[key] - final_points[key])) for key in target
        }
        return self._values(self.q)
