"""Retarget SenseGlove finger flexion onto the DG5F robot hand URDF and
visualize the result with vedo. No MuJoCo, no Perception Neuron, no arm --
just the hand geometry extracted by scripts/extract_hand_urdf.py, driven by
robot_hand/hand_retarget.py's Phase-1 flexion mapping (plan section 9).

The DG5F's *visual* mesh files aren't in this repo (see
extract_hand_urdf.py's docstring), but every finger-segment link DOES carry
a real URDF collision primitive -- the <collision><capsule radius=""
length=""/></collision> that same script authors and
robot_hand/hand_retarget.py's self-collision check uses. This viewer draws
exactly that: each capsule-bearing link is rendered as an actual capsule
(a Cylinder + two end Spheres, sized from that link's own URDF radius/
length -- vedo has no single native "capsule" primitive), so what's on
screen is the robot's own URDF geometry, not an unrelated stand-in
skeleton. Links with no authored collision (mount/base/palm/tip) fall back
to a small translucent marker sphere, purely for visual continuity.

Usage:
    python scripts/hand_retarget_vedo.py --backend mock
    python scripts/hand_retarget_vedo.py --backend mock --hand right
    python scripts/hand_retarget_vedo.py --backend bridge   # real glove, via senseglove_bridge.exe
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time
from typing import Dict, Optional

_ROOT = pathlib.Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402
import vedo  # noqa: E402

from devices.senseglove.sg_reader import (  # noqa: E402
    MockSenseGloveReader,
    SenseGloveJSONBridgeReader,
    SenseGloveReaderBase,
    SGCoreSenseGloveReader,
)
from devices.senseglove.sg_types import HandState  # noqa: E402
from robot_hand.hand_retarget import Dg5fHandRetargeter  # noqa: E402
from robot_hand.urdf_fk import UrdfTree, load_urdf  # noqa: E402
from teleop.geometry import Pose, quat_identity  # noqa: E402

_URDF_PATHS = {
    "right": _ROOT / "robot_hand" / "urdf" / "dg5f_right.urdf",
    "left": _ROOT / "robot_hand" / "urdf" / "dg5f_left.urdf",
}
_JOINT_PREFIX = {"right": "right_hand_rj_dg", "left": "left_hand_lj_dg"}
_HAND_COLOR = {"right": "cyan3", "left": "gold"}
# A finger's bones flip to this color while robot_hand.hand_retarget's
# self-collision avoidance is actively holding it back below what
# SenseGlove asked for (see HandRig.update_actors).
_CLAMP_COLOR = "red3"
# Fixed anchors so both hands are visible side by side with no arm/torso to
# place them relative to -- swap for the real wrist retargeting output
# (teleop/calibration.py's WristRetargeter target Pose) once an arm exists.
_MOUNT_ANCHOR = {"right": np.array([0.15, -0.12, 1.0]), "left": np.array([0.15, 0.12, 1.0])}


def _pose_matrix(pose: Pose) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = pose.as_matrix()
    m[:3, 3] = pose.pos
    return m


class CapsuleActor:
    """A capsule (Cylinder + 2 end Spheres) built ONCE at a fixed
    radius/length taken straight from a link's URDF <collision><capsule>,
    then repositioned every frame via an absolute rigid transform --
    radius/length never change (it's a rigid body), only its world pose
    does. `set_pose` uses transform.reset() + apply_transform(M) (vedo's
    LinearTransform.concatenate()s by default) to SET an absolute pose
    each call rather than accumulating one.
    """

    def __init__(self, radius: float, length: float, color: str):
        self.cylinder = vedo.Cylinder(r=radius, height=length, axis=(0, 0, 1), c=color)
        self.cap0 = vedo.Sphere(pos=(0, 0, -length / 2.0), r=radius, c=color)
        self.cap1 = vedo.Sphere(pos=(0, 0, length / 2.0), r=radius, c=color)
        self.assembly = vedo.Assembly([self.cylinder, self.cap0, self.cap1])

    def set_pose(self, pose: Pose) -> None:
        self.assembly.transform.reset()
        self.assembly.apply_transform(_pose_matrix(pose))

    def set_color(self, color: str) -> None:
        self.cylinder.c(color)
        self.cap0.c(color)
        self.cap1.c(color)


class HandRig:
    """Builds its vedo actors ONCE and repositions them in place every
    frame instead of recreating VTK geometry each tick -- recreating
    ~55 Sphere/Cylinder meshes per hand every frame was the actual
    bottleneck (measured ~80ms/frame "compute" at 30 Hz target -- see the
    achieved/compute/render Hz report this script prints every 2s), not
    the render call itself.
    """

    def __init__(self, side: str, color: str):
        self.side = side
        self.color = color
        self.tree: UrdfTree = load_urdf(str(_URDF_PATHS[side]))
        self.retargeter = Dg5fHandRetargeter(self.tree, _JOINT_PREFIX[side])
        self.root_pose = Pose(_MOUNT_ANCHOR[side], quat_identity())

        init_poses = self.link_poses(np.zeros(5))

        # Finger-segment links: real URDF collision capsules.
        self.capsules: Dict[str, CapsuleActor] = {}
        for link_name, link in self.tree.links.items():
            if link.collision_radius > 0.0:
                self.capsules[link_name] = CapsuleActor(link.collision_radius, link.collision_length, color)

        # Everything else (mount/base/palm/tip): no URDF collision geometry
        # was authored for these, so they get a plain marker sphere rather
        # than an invented capsule size.
        self.markers: Dict[str, "vedo.Sphere"] = {
            name: vedo.Sphere(pos=init_poses[name].pos, r=0.006, c=color).alpha(0.35)
            for name in self.tree.links
            if name not in self.capsules
        }

        self._reposition(init_poses)

    def link_poses(self, flexion: np.ndarray) -> Dict[str, Pose]:
        angles = self.retargeter.retarget(flexion)
        return self.tree.forward_kinematics(angles, self.root_pose)

    def actors(self) -> list:
        actors = list(self.markers.values())
        for capsule in self.capsules.values():
            actors.append(capsule.assembly)
        return actors

    def _reposition(self, poses: Dict[str, Pose]) -> None:
        for link_name, capsule in self.capsules.items():
            link = self.tree.links[link_name]
            capsule.set_pose(poses[link_name].compose(link.collision_origin))
        for link_name, marker in self.markers.items():
            marker.pos(poses[link_name].pos)

    def update_actors(self, flexion: np.ndarray) -> None:
        poses = self.link_poses(flexion)
        self._reposition(poses)

        # Self-collision motion limiting is otherwise invisible (it just
        # looks like the hand stopped closing) -- flip a finger's capsules
        # to _CLAMP_COLOR whenever robot_hand.hand_retarget clamped it
        # below what SenseGlove actually asked for, so it's visible that
        # the limiter is the reason, not e.g. a stale/stuck reading.
        for finger, info in self.retargeter.last_clamp.items():
            clamped = info.applied < info.requested - 1e-3
            color = _CLAMP_COLOR if clamped else self.color
            for link_name in self.retargeter.capsule_links[finger]:
                self.capsules[link_name].set_color(color)


def main():
    parser = argparse.ArgumentParser(
        description="SenseGlove -> DG5F hand URDF retargeting, visualized with vedo"
    )
    parser.add_argument("--backend", choices=["mock", "sgcore", "bridge"], default="mock")
    parser.add_argument("--hand", choices=["left", "right", "both"], default="both")
    parser.add_argument("--bridge-host", default="127.0.0.1")
    parser.add_argument("--bridge-port", type=int, default=8850)
    parser.add_argument("--hz", type=float, default=30.0, help="visual update rate")
    args = parser.parse_args()

    if args.backend == "mock":
        reader: SenseGloveReaderBase = MockSenseGloveReader()
    elif args.backend == "sgcore":
        reader = SGCoreSenseGloveReader()
    else:
        reader = SenseGloveJSONBridgeReader(args.bridge_host, args.bridge_port)
    reader.connect()

    sides = ["left", "right"] if args.hand == "both" else [args.hand]
    rigs = {side: HandRig(side, _HAND_COLOR[side]) for side in sides}
    last_hand: Dict[str, HandState] = {}

    plotter = vedo.Plotter(title="SenseGlove -> DG5F hand retargeting (vedo)", bg="white", axes=4)
    plotter.add(vedo.Grid(s=(1, 1)).c("grey8").alpha(0.2))
    for rig in rigs.values():
        plotter.add(rig.actors())

    state = {
        "n_frames": 0,
        "last_tick": None, "last_report": time.perf_counter(),
        "sum_dt": 0.0, "sum_compute": 0.0, "sum_render": 0.0, "n_since_report": 0,
    }

    def update(_evt):
        now = time.perf_counter()
        if state["last_tick"] is not None:
            state["sum_dt"] += now - state["last_tick"]
        state["last_tick"] = now

        t0 = time.perf_counter()
        left, right = reader.poll()
        if left is not None:
            last_hand["left"] = left
        if right is not None:
            last_hand["right"] = right

        n_actors = 0
        for side in sides:
            hs: Optional[HandState] = last_hand.get(side)
            flexion = hs.flexion if hs is not None else np.zeros(5)
            rigs[side].update_actors(flexion)
            n_actors += len(rigs[side].actors())
        t1 = time.perf_counter()

        plotter.render()
        t2 = time.perf_counter()

        state["sum_compute"] += t1 - t0
        state["sum_render"] += t2 - t1
        state["n_since_report"] += 1
        state["n_frames"] += 1

        if now - state["last_report"] > 2.0:
            n = state["n_since_report"]
            achieved_hz = n / (now - state["last_report"])
            print(
                f"[hand_retarget_vedo] achieved={achieved_hz:.1f} Hz (target={args.hz:.0f}) "
                f"compute={1000 * state['sum_compute'] / n:.2f} ms/frame "
                f"render={1000 * state['sum_render'] / n:.2f} ms/frame "
                f"n_actors={n_actors}"
            )
            state["last_report"] = now
            state["sum_compute"] = state["sum_render"] = 0.0
            state["n_since_report"] = 0

    print(f"[hand_retarget_vedo] backend={args.backend} hand={args.hand} -- close the window to stop")
    plotter.add_callback("timer", update)
    plotter.timer_callback("create", dt=int(1000.0 / args.hz))
    plotter.show(interactive=True)

    reader.close()
    plotter.close()


if __name__ == "__main__":
    main()
