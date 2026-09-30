"""Process 3 of 3: physics-only MuJoCo viewer.

Does NO retargeting and connects to NO tracker (mocap/Quest/SenseGlove) --
its only inputs are two UDP articulation streams (teleop/articulation_
protocol.py's ArticulationSender/Receiver): process 1 (scripts/hand_process.py,
finger joint angles) and process 2 (scripts/mocap_process.py, arm/torso joint
angles), both pointed at this process's --port. Received joint angles become
actuator ctrl targets (a servo target, not the literal pose). Loads
assets/scene/scene.xml by default (the actuated robot -- robot_hand/
mujoco_urdf.py's build_actuated_model()/actuated_mjcf_path() -- plus a
pedestal, table, and an example cloth flexcomp; --robot-only skips the
scene and loads just the robot), stepped with mj_step so gravity/contacts
are real physics, not the kinematic mj_forward-only puppet scripts/
upper_body_retarget_mujoco.py used before this split.

Run all three processes with run_all.bat, or standalone:
    python -m scripts.mujoco_physics_process --port 6100
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

_ROOT = pathlib.Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402
import numpy as np  # noqa: E402

from robot_hand.mujoco_urdf import actuated_mjcf_path  # noqa: E402
from scripts.upper_body_retarget_mujoco import (  # noqa: E402
    _HEAD_JOINTS, MujocoJointWriter, SelfCollisionMonitor,
)
from teleop.articulation_protocol import ArticulationReceiver  # noqa: E402

_URDF = _ROOT / "rby1_dg5f.urdf"
_SCENE = _ROOT / "assets" / "scene" / "scene.xml"


class CameraView:
    """Renders the model's own <camera>s (robot_hand/mujoco_urdf.py's
    cam_head/cam_left_wrist/cam_right_wrist) via mujoco.Renderer -- MuJoCo's
    own offscreen rendering, off the SAME model/data the main mujoco.viewer
    window already has, not a second copy kept in sync over IPC.
    mujoco.Renderer itself is part of MuJoCo's own rendering stack (not a
    separate 3D toolkit like vedo/Open3D, which is what actually caused a
    same-process GL conflict earlier in this project), so it can share this
    process with the passive viewer.

    Rendering and display are independent: `display=False` still renders
    every update() (populating self.last_images, e.g. for a future
    training-data-saving loop to read) without ever opening a window --
    useful for a headless data-capture run where nobody's watching but the
    frames are still wanted. `display=True` additionally opens the
    matplotlib window (TkAgg), the same "safe to run alongside
    mujoco.viewer" pattern scripts/skeleton_graph.py already established.
    """

    def __init__(
        self, model: mujoco.MjModel, data: mujoco.MjData, cam_names: list,
        width: int = 160, height: int = 120, display: bool = True,
    ):
        self.model = model
        self.data = data
        self.cam_names = cam_names
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.last_images: dict = {name: None for name in cam_names}

        self.display = display
        self._plt = None
        self.images = []
        if display:
            import matplotlib
            matplotlib.use("TkAgg")
            import matplotlib.pyplot as plt
            self._plt = plt
            plt.ion()
            self.fig, axes = plt.subplots(1, len(cam_names), num="Robot cameras", figsize=(4 * len(cam_names), 3.2))
            self.axes = [axes] if len(cam_names) == 1 else list(axes)
            for ax, name in zip(self.axes, cam_names):
                ax.set_title(name, fontsize=9)
                ax.axis("off")
                self.images.append(ax.imshow(np.zeros((height, width, 3), dtype=np.uint8)))
            self.fig.tight_layout()
            self.fig.show()

    def update(self) -> dict:
        """Always renders (into self.last_images); only touches the
        matplotlib canvas if display=True. Returns self.last_images."""
        for i, name in enumerate(self.cam_names):
            self.renderer.update_scene(self.data, camera=name)
            img = self.renderer.render()
            self.last_images[name] = img
            if self.display:
                self.images[i].set_data(img)
        if self.display:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        return self.last_images

    def close(self) -> None:
        if self._plt is not None:
            self._plt.close(self.fig)
        self.renderer.close()


def _max_tracking_error_deg(model: mujoco.MjModel, data: mujoco.MjData, writer: MujocoJointWriter):
    """Largest |ctrl - qpos| across every actuated joint, in degrees, and
    which joint it's on -- the "actuator/physics lag" half of the latency
    question: a big, persistent value here means the servo hasn't caught up
    to its target (gain too soft, load too heavy, or a genuinely fast
    target change), regardless of how fast the data got here over the
    network. Small/near-zero means the actuator is tracking fine and any
    perceived delay is upstream (network or retargeting compute -- see
    ArticulationReceiver.network_latency_ms/pipeline_latency_ms)."""
    worst_name, worst_deg = None, 0.0
    for name, ctrl_adr in writer._ctrladr.items():
        qpos_adr = writer._qposadr.get(name)
        if qpos_adr is None:
            continue
        err_deg = abs(data.ctrl[ctrl_adr] - data.qpos[qpos_adr]) * 180.0 / np.pi
        if err_deg > worst_deg:
            worst_name, worst_deg = name, err_deg
    return worst_name, worst_deg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=6100,
                         help="both hand_process.py's and mocap_process.py's --send should point here")
    parser.add_argument("--hz", type=float, default=60.0, help="physics step rate")
    parser.add_argument(
        "--diagnose-latency", action="store_true",
        help="print, once a second: network/pipeline latency from ArticulationReceiver "
        "(is the delay upstream -- network transit or retargeting compute) vs. the worst "
        "ctrl-vs-qpos tracking error across all actuated joints (is it the actuator/physics "
        "settling instead). Local-machine UDP should make network_latency_ms ~0; if the "
        "perceived delay tracks pipeline_latency_ms instead, it's compute upstream, not the "
        "network. If it tracks the tracking-error joint instead, it's actuator gain/load, not "
        "communication at all.",
    )
    parser.add_argument(
        "--robot-only", action="store_true",
        help="load robot_hand/mujoco_urdf.py's actuated robot model directly, skipping "
        "assets/scene/scene.xml's pedestal/table/cloth environment.",
    )
    parser.add_argument(
        "--show-cameras", action="store_true",
        help="open a second window showing the head + both wrist cameras "
        "(robot_hand/mujoco_urdf.py's cam_head/cam_left_wrist/cam_right_wrist), "
        "rendered via mujoco.Renderer off the same model/data as the main viewer. Implies --capture-cameras.",
    )
    parser.add_argument(
        "--capture-cameras", action="store_true",
        help="render the cameras (CameraView.last_images) without opening a window -- for a future "
        "headless training-data-capture loop to read; no visible cost if nothing reads it.",
    )
    parser.add_argument(
        "--camera-hz", type=float, default=3.0,
        help="camera render rate, independent of --hz -- measured cost of rendering all 3 cameras at "
        "160x120 is roughly proportional to pixel count (was ~370-420ms/call at 320x240), vs ~60ms for "
        "a full physics tick's substeps. Throttling only controls how OFTEN this stall happens, not its "
        "length, so keep this low; each firing will still visibly pause the physics/viewer loop.",
    )
    parser.add_argument(
        "--no-wrist-cameras", action="store_true",
        help="with --show-cameras/--capture-cameras, render only cam_head and skip cam_left_wrist/"
        "cam_right_wrist -- cuts render cost roughly to a third since it's ~linear in camera count.",
    )
    args = parser.parse_args()

    robot_mjcf = actuated_mjcf_path(_URDF)  # scene.xml's <include> needs this file to already exist on disk
    model_path = robot_mjcf if args.robot_only else _SCENE
    model = mujoco.MjModel.from_xml_path(str(model_path))
    print(f"[mujoco_physics_process] loaded {model_path}")
    data = mujoco.MjData(model)
    writer = MujocoJointWriter(model, skip=_HEAD_JOINTS)
    mujoco.mj_forward(model, data)  # baseline pose for SelfCollisionMonitor's exclude set
    collision_monitor = SelfCollisionMonitor(model, data)
    receiver = ArticulationReceiver(args.port)

    camera_view = None
    if args.show_cameras or args.capture_cameras:
        cam_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(model.ncam)
        ]
        if args.no_wrist_cameras:
            cam_names = [n for n in cam_names if n == "cam_head"]
        if not cam_names:
            print("[mujoco_physics_process] --show/capture-cameras: model has no cameras, skipping")
        else:
            camera_view = CameraView(model, data, cam_names, display=args.show_cameras)

    tick_interval = 1.0 / args.hz
    substeps = max(1, round(tick_interval / model.opt.timestep))
    warned_missing = False

    print(f"[mujoco_physics_process] listening for articulation frames on UDP :{args.port}")
    print("[mujoco_physics_process] press R to reset the scene (robot pose + cloth) to its initial state")
    print("[mujoco_physics_process] close the viewer window to stop")

    # key_callback fires on the viewer's own (background) thread, not the
    # physics-step thread below -- just flip a flag here and apply the
    # actual reset from the main loop, so it can't race a concurrent
    # mj_step. mj_resetData restores qpos to the compiled model's own
    # qpos0 (or a <key> keyframe if one existed, which this model doesn't
    # have) -- that includes the cloth's flex vertices, since those are
    # ordinary generalized coordinates too, not something separate.
    reset_requested = [False]

    def key_callback(keycode: int) -> None:
        if keycode in (ord("R"), ord("r")):
            reset_requested[0] = True

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        # Fixed viewpoint: no head DOF on the real robot means no reason for
        # the viewer to orbit/follow anything either.
        viewer.cam.lookat[:] = [0.0, 0.0, 1.0]
        viewer.cam.distance = 2.5
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -15

        # Hide collision-only geom groups (coarse capsule/STL shapes) so
        # only the detailed visual .obj meshes show -- see
        # scripts/upper_body_retarget_mujoco.py's identical block for why.
        # Only groups 0/1 are the robot's own collision/visual split
        # (mujoco_urdf.py's URDF import convention); scene furniture
        # (assets/scene/scene.xml) is deliberately on group 2+ so it's
        # never swept into this toggle -- confirmed the hard way: an
        # earlier scene.xml left table/pedestal geoms on the default group
        # (0), which put them in the same bucket as the robot's collision
        # capsules and hid them outright.
        collision_only = [True, True]
        for i in range(model.ngeom):
            group = model.geom_group[i]
            if 0 <= group < len(collision_only):
                if model.geom_contype[i] == 0 and model.geom_conaffinity[i] == 0:
                    collision_only[group] = False
        for group, is_collision_only in enumerate(collision_only):
            if is_collision_only:
                viewer.opt.geomgroup[group] = 0

        last_tick = 0.0
        last_diag = 0.0
        last_camera = 0.0
        camera_interval = 1.0 / args.camera_hz
        perf_ticks = 0
        perf_window_start = time.monotonic()
        last_camera_render_ms = 0.0
        while viewer.is_running():
            now = time.monotonic()
            if now - last_tick < tick_interval:
                time.sleep(max(0.0, tick_interval - (now - last_tick)))
                continue
            last_tick = now

            if reset_requested[0]:
                reset_requested[0] = False
                mujoco.mj_resetData(model, data)
                mujoco.mj_forward(model, data)
                print("[mujoco_physics_process] reset to initial state")

            receiver.poll()
            missing = writer.write_ctrl(data, receiver.joints)
            if missing and not warned_missing:
                warned_missing = True
                print(
                    f"[mujoco_physics_process] WARNING: {missing} received joint name(s) have no "
                    "matching MuJoCo actuator this tick -- printed once"
                )

            for _ in range(substeps):
                mujoco.mj_step(model, data)
            collision_monitor.check(data, now, warn=False)
            viewer.sync()
            if camera_view is not None and now - last_camera >= camera_interval:
                last_camera = now
                t_render0 = time.perf_counter()
                camera_view.update()
                last_camera_render_ms = (time.perf_counter() - t_render0) * 1000.0

            if args.diagnose_latency and now - last_diag >= 1.0:
                last_diag = now
                worst_name, worst_deg = _max_tracking_error_deg(model, data, writer)
                print(
                    f"[latency] network={receiver.network_latency_ms:.1f}ms "
                    f"pipeline={receiver.pipeline_latency_ms:.1f}ms "
                    f"| tracking_error worst={worst_name}:{worst_deg:.1f}deg"
                    if receiver.network_latency_ms is not None else
                    "[latency] no frames received yet"
                )

            # Real-time-ness: how close the achieved loop rate is to --hz,
            # printed once a second regardless of --diagnose-latency (this
            # is about the physics/viewer loop's own pacing, not the
            # network/actuator question that flag covers). A big gap below
            # --hz means something in the tick (physics substeps, camera
            # render, viewer.sync) is taking longer than tick_interval
            # allows, not that anything is "waiting" -- there's no sleep
            # left to give back in that case.
            perf_ticks += 1
            if now - perf_window_start >= 1.0:
                achieved_hz = perf_ticks / (now - perf_window_start)
                cam_note = f", last_camera_render={last_camera_render_ms:.0f}ms" if camera_view is not None else ""
                print(
                    f"[perf] target_hz={args.hz:.0f} achieved_hz={achieved_hz:.1f}{cam_note}"
                )
                perf_ticks = 0
                perf_window_start = now

    receiver.close()
    if camera_view is not None:
        camera_view.close()


if __name__ == "__main__":
    main()
