"""Local, no-MuJoCo sanity-check visualizer for the device PC.

Run this on the device PC (Perception Neuron + SenseGlove attached) to
confirm the data actually looks right *before* wiring up the UDP link and
the MuJoCo receiver on a second machine (plan section 22: "이 두 데이터가
안정적으로 나오면 이후 MuJoCo 쪽 통합 작업으로 넘어간다"). Uses vedo (a thin,
fast-to-install VTK wrapper) instead of MuJoCo, so this script has zero
dependency on the MuJoCo PC, the network, or mujoco_teleop/*.

Shows raw (un-retargeted, un-calibrated) chest/wrist poses as RGB axis
triads and raw SenseGlove finger joints as points -- exactly the same
values scripts/teleop_sender.py would read and send, just visualized
locally first.

Usage:
    python scripts/vedo_local_preview.py --backend mock
    python scripts/vedo_local_preview.py --backend real
"""

from __future__ import annotations

import argparse
import pathlib
import sys

_ROOT = pathlib.Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402
import vedo  # noqa: E402

from devices.perception_neuron.pn_reader import MocapApiPNReader, MockPNReader  # noqa: E402
from devices.senseglove.sg_reader import MockSenseGloveReader, SGCoreSenseGloveReader  # noqa: E402
from teleop.calibration import map_hand_points_to_world  # noqa: E402
from teleop.geometry import Pose  # noqa: E402
from teleop.teleop_state import HumanTeleopState, TeleopAggregator  # noqa: E402

_AXIS_COLORS = ("red4", "green4", "blue4")


def pose_axis_actors(pose: Pose, length: float = 0.08, radius: float = 0.004) -> list:
    """RGB axis triad (X/Y/Z) + a small white sphere at the origin, for one pose."""
    endpoints = pose.axis_endpoints(length)
    actors = [
        vedo.Cylinder(pos=[pose.pos, endpoints[i]], r=radius, c=_AXIS_COLORS[i])
        for i in range(3)
    ]
    actors.append(vedo.Sphere(pos=pose.pos, r=radius * 1.8, c="white"))
    return actors


def hand_point_actors(wrist_pose: Pose, joint_positions_local: np.ndarray, color: str) -> list:
    world_pts = map_hand_points_to_world(wrist_pose, joint_positions_local)
    return [vedo.Points(world_pts.reshape(-1, 3), r=12, c=color)]


def build_scene_actors(state: HumanTeleopState) -> list:
    actors = []
    actors += pose_axis_actors(state.chest, length=0.1, radius=0.006)
    actors += pose_axis_actors(state.left_wrist, length=0.07)
    actors += pose_axis_actors(state.right_wrist, length=0.07)
    actors += hand_point_actors(state.left_wrist, state.left_hand.joint_positions, "gold")
    actors += hand_point_actors(state.right_wrist, state.right_hand.joint_positions, "cyan3")
    return actors


def main():
    parser = argparse.ArgumentParser(description="Local vedo sanity-check visualizer (no MuJoCo/UDP)")
    parser.add_argument("--backend", choices=["mock", "real"], default="mock")
    parser.add_argument("--hz", type=float, default=30.0, help="visual update rate")
    args = parser.parse_args()

    if args.backend == "mock":
        pn_reader = MockPNReader()
        sg_reader = MockSenseGloveReader()
    else:
        pn_reader = MocapApiPNReader()
        sg_reader = SGCoreSenseGloveReader()

    aggregator = TeleopAggregator(pn_reader, sg_reader)
    aggregator.connect()

    plotter = vedo.Plotter(title="PN + SenseGlove raw data preview (local, no MuJoCo)", bg="white", axes=4)
    ground = vedo.Grid(s=(2, 2)).c("grey8").alpha(0.25)
    plotter.add(ground)

    state = {"actors": [], "n_frames": 0}

    def update(_event):
        teleop_state = aggregator.poll()
        if teleop_state is None:
            return
        if state["actors"]:
            plotter.remove(state["actors"])
        state["actors"] = build_scene_actors(teleop_state)
        plotter.add(state["actors"])
        plotter.render()
        state["n_frames"] += 1
        if state["n_frames"] % 150 == 0:
            print(
                f"[vedo_local_preview] frames={state['n_frames']} "
                f"pn_age={teleop_state.pn_age:.3f}s sg_age={teleop_state.sg_age:.3f}s"
            )

    print(f"[vedo_local_preview] backend={args.backend} -- close the window to stop")
    plotter.add_callback("timer", update)
    plotter.timer_callback("create", dt=int(1000.0 / args.hz))
    plotter.show(interactive=True)

    aggregator.close()
    plotter.close()


if __name__ == "__main__":
    main()
