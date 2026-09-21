"""Retarget SenseGlove finger flexion onto the DG5F robot hand URDF and
visualize the result with vedo. No MuJoCo, no Perception Neuron, no arm --
just the hand geometry extracted by scripts/extract_hand_urdf.py, driven by
robot_hand/hand_retarget.py's Phase-1 flexion mapping (plan section 9).

Two render modes (--render):
  mesh (default) -- every link's actual URDF <visual><mesh> (a .dae under
    the repo-root dg5f/, loaded by robot_hand/mesh_loader.py), in that
    part's own material colors. This is the real DG5F geometry, not a
    stand-in -- heavier to render than capsule mode (see the known-issue
    Hz note in RUN_GUIDE.md).
  capsule -- every finger-segment link's <collision><capsule radius=""
    length=""/></collision> (the same primitive robot_hand/hand_retarget.py's
    self-collision check uses) drawn as a Cylinder + two end Spheres (vedo
    has no single native "capsule" primitive). Links with no authored
    collision (mount/base/palm/tip) fall back to a small translucent
    marker sphere. Cheaper to render; useful when mesh mode's frame rate
    is too low to be usable.

In both modes, a finger's links flip to red while
robot_hand/hand_retarget.py's self-collision avoidance is actively holding
it back below what SenseGlove asked for.

Usage:
    python scripts/hand_retarget_vedo.py --backend mock
    python scripts/hand_retarget_vedo.py --backend mock --hand right
    python scripts/hand_retarget_vedo.py --backend mock --render capsule
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
from robot_hand import mesh_loader  # noqa: E402
from robot_hand.hand_retarget import Dg5fHandRetargeter  # noqa: E402
from robot_hand.urdf_fk import UrdfLink, UrdfTree, load_urdf  # noqa: E402
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

    Every actor class here (CapsuleActor, MeshActor, MarkerActor) exposes
    the same three members so HandRig can treat all of a hand's links
    uniformly: `.actor` (the vedo object to add to the plotter),
    `.local_origin` (this link's Pose to compose onto its FK world pose
    before positioning), `.set_pose(Pose)`, and `.set_color(Optional[str])`
    (None means "back to this part's own resting color").
    """

    def __init__(self, radius: float, length: float, color: str, local_origin: Pose):
        self.color = color
        self.local_origin = local_origin
        self.cylinder = vedo.Cylinder(r=radius, height=length, axis=(0, 0, 1), c=color)
        self.cap0 = vedo.Sphere(pos=(0, 0, -length / 2.0), r=radius, c=color)
        self.cap1 = vedo.Sphere(pos=(0, 0, length / 2.0), r=radius, c=color)
        self.actor = vedo.Assembly([self.cylinder, self.cap0, self.cap1])

    def set_pose(self, pose: Pose) -> None:
        self.actor.transform.reset()
        self.actor.apply_transform(_pose_matrix(pose))

    def set_color(self, color: Optional[str]) -> None:
        c = color if color is not None else self.color
        self.cylinder.c(c)
        self.cap0.c(c)
        self.cap1.c(c)


class MarkerActor:
    """Fallback for links with no authored collision/visual geometry: a
    small translucent marker sphere, purely for visual continuity.

    Wrapped in a single-element vedo.Assembly (like CapsuleActor/MeshActor)
    rather than calling apply_transform directly on the bare Sphere --
    vedo's Points.apply_transform mishandles a raw 4x4 ndarray on a plain
    Mesh/Points object (`not LT` on a multi-element ndarray raises
    "truth value ... ambiguous"); Assembly.apply_transform doesn't hit
    that path.
    """

    def __init__(self, radius: float, color: str, local_origin: Pose):
        self.color = color
        self.local_origin = local_origin
        self.sphere = vedo.Sphere(r=radius, c=color).alpha(0.35)
        self.actor = vedo.Assembly([self.sphere])

    def set_pose(self, pose: Pose) -> None:
        self.actor.transform.reset()
        self.actor.apply_transform(_pose_matrix(pose))

    def set_color(self, color: Optional[str]) -> None:
        self.sphere.c(color if color is not None else self.color).alpha(0.35)


class MeshActor:
    """This link's real URDF <visual><mesh> (robot_hand/mesh_loader.py),
    built ONCE and repositioned every frame like CapsuleActor. A .dae can
    bundle several differently-colored sub-meshes (e.g. the palm), so this
    keeps each sub-mesh's own (part, original_rgba) to restore after a
    clamp-color override -- set_color(None) puts every part's own material
    color back rather than flattening the whole link to one hand color.
    """

    def __init__(self, urdf_filename: str, local_origin: Pose):
        self.local_origin = local_origin
        self.parts = mesh_loader.load_mesh_parts(urdf_filename)
        for mesh, rgba in self.parts:
            mesh.color((rgba[0] / 255.0, rgba[1] / 255.0, rgba[2] / 255.0)).alpha(rgba[3] / 255.0)
        self.actor = vedo.Assembly([mesh for mesh, _ in self.parts])

    def set_pose(self, pose: Pose) -> None:
        self.actor.transform.reset()
        self.actor.apply_transform(_pose_matrix(pose))

    def set_color(self, color: Optional[str]) -> None:
        for mesh, rgba in self.parts:
            if color is None:
                mesh.color((rgba[0] / 255.0, rgba[1] / 255.0, rgba[2] / 255.0)).alpha(rgba[3] / 255.0)
            else:
                mesh.color(color).alpha(1.0)


class HandRig:
    """Builds its vedo actors ONCE and repositions them in place every
    frame instead of recreating VTK geometry each tick -- recreating
    ~55 Sphere/Cylinder meshes per hand every frame was the actual
    bottleneck (measured ~80ms/frame "compute" at 30 Hz target -- see the
    achieved/compute/render Hz report this script prints every 2s), not
    the render call itself.
    """

    def __init__(self, side: str, color: str, render_mode: str = "mesh"):
        self.side = side
        self.color = color
        self.render_mode = render_mode
        self.tree: UrdfTree = load_urdf(str(_URDF_PATHS[side]))
        self.retargeter = Dg5fHandRetargeter(self.tree, _JOINT_PREFIX[side])
        self.root_pose = Pose(_MOUNT_ANCHOR[side], quat_identity())

        init_poses = self.link_poses(np.zeros(5))

        self.link_actors: Dict[str, object] = {
            name: self._make_actor(link, color) for name, link in self.tree.links.items()
        }
        self._reposition(init_poses)

    def _make_actor(self, link: UrdfLink, color: str):
        if self.render_mode == "mesh" and link.visual_mesh:
            return MeshActor(link.visual_mesh, link.visual_origin)
        if link.collision_radius > 0.0:
            return CapsuleActor(link.collision_radius, link.collision_length, color, link.collision_origin)
        # Links with no capsule and (in capsule mode) no mesh either --
        # mount/base/palm/tip -- fall back to a plain marker sphere.
        return MarkerActor(0.006, color, Pose.identity())

    def link_poses(self, flexion: np.ndarray) -> Dict[str, Pose]:
        angles = self.retargeter.retarget(flexion)
        return self.tree.forward_kinematics(angles, self.root_pose)

    def actors(self) -> list:
        return [actor.actor for actor in self.link_actors.values()]

    def _reposition(self, poses: Dict[str, Pose]) -> None:
        for link_name, actor in self.link_actors.items():
            actor.set_pose(poses[link_name].compose(actor.local_origin))

    def update_actors(self, flexion: np.ndarray) -> None:
        poses = self.link_poses(flexion)
        self._reposition(poses)

        # Self-collision motion limiting is otherwise invisible (it just
        # looks like the hand stopped closing) -- flip a finger's links to
        # _CLAMP_COLOR whenever robot_hand.hand_retarget clamped it below
        # what SenseGlove actually asked for, so it's visible that the
        # limiter is the reason, not e.g. a stale/stuck reading. `None`
        # restores each actor's own resting color (the hand's flat color
        # for capsule/marker, this part's real material color for mesh).
        for finger, info in self.retargeter.last_clamp.items():
            clamped = info.applied < info.requested - 1e-3
            color = _CLAMP_COLOR if clamped else None
            for link_name in self.retargeter.capsule_links[finger]:
                self.link_actors[link_name].set_color(color)


def main():
    parser = argparse.ArgumentParser(
        description="SenseGlove -> DG5F hand URDF retargeting, visualized with vedo"
    )
    parser.add_argument("--backend", choices=["mock", "sgcore", "bridge"], default="mock")
    parser.add_argument("--hand", choices=["left", "right", "both"], default="both")
    parser.add_argument("--render", choices=["mesh", "capsule"], default="mesh")
    parser.add_argument("--bridge-host", default="127.0.0.1")
    parser.add_argument("--bridge-port", type=int, default=8850)
    parser.add_argument("--hz", type=float, default=30.0, help="poll + retarget tick rate")
    parser.add_argument(
        "--render-hz", type=float, default=30.0,
        help="capped vedo redraw rate, decoupled from --hz so a fast poll loop "
             "doesn't push every tick into the (expensive) VTK render call",
    )
    args = parser.parse_args()

    if args.backend == "mock":
        reader: SenseGloveReaderBase = MockSenseGloveReader()
    elif args.backend == "sgcore":
        reader = SGCoreSenseGloveReader()
    else:
        reader = SenseGloveJSONBridgeReader(args.bridge_host, args.bridge_port)
    reader.connect()

    sides = ["left", "right"] if args.hand == "both" else [args.hand]
    rigs = {side: HandRig(side, _HAND_COLOR[side], args.render) for side in sides}
    last_hand: Dict[str, HandState] = {}

    plotter = vedo.Plotter(title="SenseGlove -> DG5F hand retargeting (vedo)", bg="white", axes=4)
    plotter.add(vedo.Grid(s=(1, 1)).c("grey8").alpha(0.2))
    for rig in rigs.values():
        plotter.add(rig.actors())

    render_interval = 1.0 / args.render_hz
    state = {
        "n_frames": 0,
        "last_tick": None, "last_report": time.perf_counter(),
        "sum_dt": 0.0, "sum_compute": 0.0, "sum_render": 0.0, "n_since_report": 0,
        "last_render": 0.0, "n_rendered": 0,
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

        # plotter.render() is the expensive VTK draw call (confirmed by the
        # render= timing below) — gate it to render_hz independent of the
        # poll/retarget tick rate, same fix as the mocap hand raw plot: don't
        # push every tick into the renderer, the eye can't tell the
        # difference above ~30Hz anyway.
        if t1 - state["last_render"] >= render_interval:
            plotter.render()
            state["last_render"] = t1
            state["n_rendered"] += 1
        t2 = time.perf_counter()

        state["sum_compute"] += t1 - t0
        state["sum_render"] += t2 - t1
        state["n_since_report"] += 1
        state["n_frames"] += 1

        if now - state["last_report"] > 2.0:
            n = state["n_since_report"]
            achieved_hz = n / (now - state["last_report"])
            render_hz = state["n_rendered"] / (now - state["last_report"])
            print(
                f"[hand_retarget_vedo] poll={achieved_hz:.1f} Hz (target={args.hz:.0f}) "
                f"render={render_hz:.1f} Hz (target={args.render_hz:.0f}) "
                f"compute={1000 * state['sum_compute'] / n:.2f} ms/frame "
                f"render_cost={1000 * state['sum_render'] / n:.2f} ms/frame "
                f"n_actors={n_actors}"
            )
            state["last_report"] = now
            state["sum_compute"] = state["sum_render"] = 0.0
            state["n_since_report"] = state["n_rendered"] = 0

    print(
        f"[hand_retarget_vedo] backend={args.backend} hand={args.hand} render={args.render} "
        "-- close the window to stop"
    )
    plotter.add_callback("timer", update)
    plotter.timer_callback("create", dt=int(1000.0 / args.hz))
    plotter.show(interactive=True)

    reader.close()
    plotter.close()


if __name__ == "__main__":
    main()
