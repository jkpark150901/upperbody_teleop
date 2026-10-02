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
import json
import os
import pathlib
import queue
import sys
import threading
import time
from typing import Optional

_ROOT = pathlib.Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _gl_backend_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--gl-backend", choices=["auto", "egl", "glfw", "osmesa", "wgl"], default="auto",
        help="OpenGL backend for CameraView's mujoco.Renderer (sets MUJOCO_GL). auto = configs/"
        "sim_params.yaml's camera.gl_backend if set to something other than 'auto' there, else egl "
        "on Linux (headless NVIDIA GPU context -- glfw there lands on whatever the X display offers, "
        "which on a server-GPU box is Mesa llvmpipe, i.e. CPU rendering, ~1s per 3-camera frame vs "
        "~7ms on EGL), wgl on Windows (confirmed safe to create from CameraView's background render "
        "thread alongside the main viewer's own GLFW window -- plain glfw there is NOT: two GLFW "
        "window-class registrations from different threads crash with 'Win32: Failed to register "
        "window class', which is why CameraView falls back to synchronous main-thread rendering "
        "whenever MUJOCO_GL isn't one of the confirmed-safe backends -- osmesa/egl/wgl, see its own "
        "threaded flag), left to mujoco's own default elsewhere (macOS). An explicit MUJOCO_GL env "
        "var wins over everything. Only affects the camera renderer -- the passive viewer always "
        "uses its own GLFW window regardless.",
    )
    parser.add_argument(
        "--egl-device", type=int, default=None,
        help="with the egl backend: GPU index to render on (sets MUJOCO_EGL_DEVICE_ID; default: "
        "configs/sim_params.yaml's camera.egl_device, else the first usable device).",
    )
    parser.add_argument(
        "--sim-config", type=pathlib.Path, default=None,
        help="path to the sim_params.yaml-shaped config (actuator gains, cloth physics, camera "
        "resolution/rate/backend). Defaults to configs/sim_params.yaml. Re-read fresh on every run. "
        "(Declared here too, not just in main()'s parser, so --gl-backend/--egl-device's own config "
        "fallback can honor a non-default --sim-config path during the pre-import GL setup below.)",
    )


def _configure_gl_backend() -> None:
    """Must run before `import mujoco` -- mujoco.Renderer's GL context
    picks its backend from MUJOCO_GL at import time, so this pre-parses
    just --gl-backend/--egl-device/--sim-config out of argv (main()'s
    parser declares them again for --help)."""
    pre = argparse.ArgumentParser(add_help=False)
    _gl_backend_args(pre)
    args, _ = pre.parse_known_args()

    config_backend, config_egl_device = "auto", None
    try:
        import yaml
        config_path = args.sim_config or (_ROOT / "configs" / "sim_params.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            cam_cfg = (yaml.safe_load(f) or {}).get("camera", {})
        config_backend = cam_cfg.get("gl_backend", "auto")
        config_egl_device = cam_cfg.get("egl_device")
    except FileNotFoundError:
        pass  # main()'s own load_sim_config() raises its own clearer error later

    backend = args.gl_backend if args.gl_backend != "auto" else config_backend
    if backend == "auto":
        if "MUJOCO_GL" in os.environ:
            backend = None
        elif sys.platform.startswith("linux"):
            backend = "egl"
        elif sys.platform == "win32":
            backend = "wgl"
        else:
            backend = None
    if backend is not None:
        os.environ["MUJOCO_GL"] = backend
        if backend in ("egl", "osmesa"):
            os.environ["PYOPENGL_PLATFORM"] = backend

    egl_device = args.egl_device if args.egl_device is not None else config_egl_device
    if egl_device is not None:
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(egl_device)


_configure_gl_backend()

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402
import numpy as np  # noqa: E402

from robot_hand.mujoco_urdf import actuated_mjcf_path  # noqa: E402
from robot_hand.sim_config import actuator_gain_kwargs, generate_cloth_xml, load_sim_config  # noqa: E402
from robot_hand.upper_body_retarget import RBY1UpperBodyRetargeter  # noqa: E402
from robot_hand.urdf_fk import load_urdf  # noqa: E402
from scripts.hand_process import _WRIST_TARGET_NAMES, _WRIST_TARGET_ROT_NAMES  # noqa: E402
from scripts.upper_body_retarget_mujoco import (  # noqa: E402
    _HEAD_JOINTS, MujocoJointWriter, SelfCollisionMonitor, _quest_forward_reference,
)
from teleop.articulation_protocol import ArticulationReceiver  # noqa: E402

_URDF = _ROOT / "rby1_dg5f.urdf"
_SCENE = _ROOT / "assets" / "scene" / "scene.xml"

_PINCH_FORCE_BODY_CANDIDATES = {
    "left": (
        ("left_hand_ll_dg_1_tip", "left_hand_ll_dg_1_4"),
        ("left_hand_ll_dg_2_tip", "left_hand_ll_dg_2_4"),
    ),
    "right": (
        ("right_hand_rl_dg_1_tip", "right_hand_rl_dg_1_4"),
        ("right_hand_rl_dg_2_tip", "right_hand_rl_dg_2_4"),
    ),
}
_PINCH_FORCE_JOINT_PREFIXES = {
    "left": ("left_hand_lj_dg_1_", "left_hand_lj_dg_2_"),
    "right": ("right_hand_rj_dg_1_", "right_hand_rj_dg_2_"),
}


def _forward_reach_qpos() -> dict:
    """Both arms extended ~90deg straight forward -- RBY1UpperBodyRetargeter's
    own Quest-calibration reference pose (see scripts/upper_body_retarget_
    mujoco.py's _quest_forward_reference), reused here as this process's
    initial pose and R-key reset target instead of qpos0 (every joint at
    0, a limp hanging-arms pose that's not useful to look at or start
    teleop from). This is the SAME pose every Quest teleop session already
    anchors its own wrist deltas to (base_wrist/base_rotation), so it's
    already the project's de facto "arms forward" reference, not a new
    pose invented here.

    _quest_forward_reference uses the same closed-form wrist-position solve
    as live Quest tracking. The older DLS-only pose could visually droop
    into a low local minimum, which made the captured "arms forward" pose
    disagree with the robot-side reference. Checked here against the HARD
    geometric limit (100% of shoulder-elbow + elbow-wrist, no live-tracking
    margin) since this pose is static, not something being continuously
    re-solved: if even that's exceeded, that side falls back to neutral
    instead of silently using a pose beyond what the arm can physically
    reach.
    """
    tree = load_urdf(str(_URDF))
    retargeter = RBY1UpperBodyRetargeter(tree, calibration_pose="attention")
    rest_poses = retargeter.poses()
    hard_limit = {}
    for side in ("left", "right"):
        shoulder = rest_poses[f"link_{side}_arm_0"].pos
        elbow = rest_poses[f"link_{side}_arm_3"].pos
        wrist = rest_poses[f"link_{side}_arm_6"].pos
        hard_limit[side] = float(np.linalg.norm(elbow - shoulder) + np.linalg.norm(wrist - elbow))

    _quest_forward_reference(retargeter)  # leaves retargeter.q at the forward-reach solution
    poses = retargeter.poses()
    values = retargeter._values(retargeter.q)
    reference_values = retargeter._values(retargeter.reference_q)
    out = {}
    for side in ("left", "right"):
        shoulder = poses[f"link_{side}_arm_0"].pos
        wrist = poses[f"link_{side}_arm_6"].pos
        reach = float(np.linalg.norm(wrist - shoulder))
        side_names = [f"{side}_arm_{i}" for i in range(7)]
        if reach > hard_limit[side]:
            print(
                f"[mujoco_physics_process] WARNING: forward-reach init pose for {side} arm exceeds "
                f"that arm's max reach ({reach:.3f}m > {hard_limit[side]:.3f}m) -- using the neutral "
                "pose for that arm instead"
            )
            out.update({n: reference_values[n] for n in side_names})
        else:
            out.update({n: values[n] for n in side_names})
    out["torso_hp"] = values["torso_hp"]
    out["torso_5"] = values["torso_5"]
    return out


def _first_body_id(model: mujoco.MjModel, names: tuple[str, ...]) -> tuple[int, str] | tuple[None, None]:
    for name in names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id >= 0:
            return body_id, name
    return None, None


def _pinch_force_pairs(model: mujoco.MjModel) -> dict[str, tuple[int, int]]:
    pairs: dict[str, tuple[int, int]] = {}
    for side, (thumb_candidates, index_candidates) in _PINCH_FORCE_BODY_CANDIDATES.items():
        thumb_id, thumb_name = _first_body_id(model, thumb_candidates)
        index_id, index_name = _first_body_id(model, index_candidates)
        if thumb_id is None or index_id is None:
            print(
                f"[mujoco_physics_process] pinch force disabled for {side}: "
                f"missing thumb/index body candidates {thumb_candidates} / {index_candidates}"
            )
            continue
        pairs[side] = (thumb_id, index_id)
        print(f"[mujoco_physics_process] pinch force {side}: {thumb_name} <-> {index_name}")
    return pairs


def _pinch_force_dofs(model: mujoco.MjModel) -> dict[str, np.ndarray]:
    dofs: dict[str, np.ndarray] = {}
    for side, prefixes in _PINCH_FORCE_JOINT_PREFIXES.items():
        side_dofs = []
        for joint_id in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if name and any(name.startswith(prefix) for prefix in prefixes):
                dof_adr = int(model.jnt_dofadr[joint_id])
                if dof_adr >= 0:
                    side_dofs.append(dof_adr)
        dofs[side] = np.asarray(side_dofs, dtype=int)
    return dofs


def _apply_pinch_contact_forces(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    pairs: dict[str, tuple[int, int]],
    dofs: dict[str, np.ndarray],
    active: dict[str, bool],
    *,
    force_n: float,
    enter_m: float,
    exit_m: float,
    mode: str,
) -> None:
    data.xfrc_applied[:] = 0.0
    data.qfrc_applied[:] = 0.0
    if force_n <= 0.0:
        return
    jacp_thumb = np.zeros((3, model.nv))
    jacr_thumb = np.zeros((3, model.nv))
    jacp_index = np.zeros((3, model.nv))
    jacr_index = np.zeros((3, model.nv))
    for side, (thumb_id, index_id) in pairs.items():
        delta = data.xpos[index_id] - data.xpos[thumb_id]
        dist = float(np.linalg.norm(delta))
        was_active = active.get(side, False)
        is_active = dist <= (exit_m if was_active else enter_m)
        active[side] = is_active
        if not is_active or dist < 1e-6:
            continue
        direction = delta / dist
        force = force_n * direction
        if mode == "body-force":
            data.xfrc_applied[thumb_id, :3] += force
            data.xfrc_applied[index_id, :3] -= force
            continue
        jacp_thumb.fill(0.0)
        jacr_thumb.fill(0.0)
        jacp_index.fill(0.0)
        jacr_index.fill(0.0)
        mujoco.mj_jacBody(model, data, jacp_thumb, jacr_thumb, thumb_id)
        mujoco.mj_jacBody(model, data, jacp_index, jacr_index, index_id)
        qfrc = jacp_thumb.T @ force - jacp_index.T @ force
        side_dofs = dofs.get(side)
        if side_dofs is None or len(side_dofs) == 0:
            data.qfrc_applied[:] += qfrc
        else:
            data.qfrc_applied[side_dofs] += qfrc[side_dofs]


class CameraView:
    """Renders the model's own <camera>s (robot_hand/mujoco_urdf.py's
    cam_head/cam_left_wrist/cam_right_wrist) via mujoco.Renderer -- MuJoCo's
    own offscreen rendering, off a SEPARATE MjData (its own private copy,
    not the main viewer loop's) kept roughly in sync with it, not a second
    model/process kept in sync over IPC. mujoco.Renderer itself is part of
    MuJoCo's own rendering stack (not a separate 3D toolkit like vedo/
    Open3D, which is what actually caused a same-process GL conflict
    earlier in this project), so it can share this process with the
    passive viewer.

    Rendering runs on its own background thread IF the GL backend is safe
    for that (osmesa/egl -- headless, no window) -- see threaded below.
    Measured cost is ~100-400ms per render() call (CPU/software rendering,
    3 cameras), vs ~1-20ms for a physics tick's own substeps, so calling
    it inline stalls mj_step/viewer.sync() for however long the render
    took, every time it fires, no matter how low --camera-hz is set.
    (That cost was the software rasterizer: on Linux the default glfw
    backend lands on Mesa llvmpipe when the X display isn't driven by the
    NVIDIA GPU. --gl-backend egl -- the Linux default now -- renders on
    the GPU instead, ~7ms for the same 3 cameras; _report_gl_renderer()
    prints which one you got.)

    threaded=True (osmesa/egl/wgl -- _configure_gl_backend() defaults
    MUJOCO_GL to one of these automatically now, egl on Linux, wgl on
    Windows, so this is the default everywhere except macOS/an explicit
    MUJOCO_GL=glfw): the render thread (1) copies the main loop's qpos/
    qvel into its OWN MjData under a short lock (cheap, just a numpy copy
    -- concurrently reading the main loop's live MjData while mj_step
    writes to it would be a data race), (2) mj_forward()s that copy
    (kinematics only, cheap, recomputes cam_xpos/xmat from the copied
    qpos), (3) renders from it with its own Renderer instance (also
    private to this thread -- mujoco.Renderer isn't meant to be shared
    across threads), (4) stores the resulting images under the same lock.
    The main loop's only job (via sync_display()) is to copy whatever's
    already rendered onto the matplotlib canvas -- cheap, no render() call
    on that path, so it doesn't block physics regardless of how slow
    rendering is.

    threaded=False (plain glfw): confirmed the hard way that a background-
    thread mujoco.Renderer blows up with "Win32: Failed to register
    window class" -- GLFW is NOT thread-safe, and mujoco.viewer.
    launch_passive's own window (always GLFW, regardless of MUJOCO_GL) is
    already live on the main thread, so a second GLFW window class
    registration from a different thread races it (wgl -- Windows' native
    GL API -- was confirmed NOT to have this problem, which is why it's
    the new Windows default instead of glfw). Falls back to rendering
    inline on whichever thread calls sync_display(), throttled to
    --camera-hz same as before -- physics DOES stall for the ~100-400ms
    render whenever it fires in this mode; accepted as unavoidable on this
    backend rather than crashing.

    `display=False` skips the matplotlib window; sync_display() still
    renders into self.last_images (e.g. for a future training-data-saving
    loop to read) for a headless capture run.
    """

    def __init__(
        self, model: mujoco.MjModel, data: mujoco.MjData, cam_names: list,
        width: int = 160, height: int = 120, display: bool = True, hz: float = 3.0,
    ):
        self.model = model
        self.main_data = data
        self.cam_names = cam_names
        self.last_images: dict = {name: None for name in cam_names}
        # The qpos/ctrl the *currently stored* last_images were actually
        # rendered from, plus a sequence number that only advances when a
        # new render batch lands -- a saver (see TeachingDataRecorder)
        # that wants a sample where the images and the joint values are
        # from the same instant reads these three together (get_sample()),
        # instead of pairing last_images with whatever data.qpos/ctrl is
        # live *right now* (which has moved on by however stale the
        # images are -- rendering is slow, physics isn't).
        self.last_qpos = data.qpos.copy()
        self.last_ctrl = data.ctrl.copy()
        self.last_sim_time: Optional[float] = None
        self.frame_seq = 0
        self._images_lock = threading.Lock()
        self._qpos_lock = threading.Lock()
        self._qpos_snapshot = data.qpos.copy()
        self._qvel_snapshot = data.qvel.copy()
        self._ctrl_snapshot = data.ctrl.copy()
        self._sim_time_snapshot = 0.0
        self._stop = threading.Event()
        self._pending_display: dict = {}

        # osmesa/egl are headless (no window of their own); wgl is
        # Windows' native GL API, confirmed safe to create a context with
        # from a background thread alongside the main viewer's own GLFW
        # window (no window-class conflict, unlike plain glfw -- which
        # IS what crashed when this was tried the first time, see class
        # docstring). _configure_gl_backend() defaults MUJOCO_GL to one
        # of these three automatically now, so this is threaded by
        # default on Linux/Windows; only an explicit MUJOCO_GL=glfw (or
        # macOS, left at mujoco's own default) falls back to synchronous.
        self.threaded = os.environ.get("MUJOCO_GL") in ("osmesa", "egl", "wgl")

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

        self._thread = None
        if self.threaded:
            self._thread = threading.Thread(
                target=self._render_loop, args=(width, height, hz), daemon=True,
            )
            self._thread.start()
        else:
            print(
                "[mujoco_physics_process] CameraView: MUJOCO_GL is not osmesa/egl (GLFW is not "
                "thread-safe), rendering inline on the main loop instead of a background thread -- "
                "physics will stall for each render's duration. Use --gl-backend osmesa/egl to avoid this."
            )
            self._renderer = mujoco.Renderer(model, height=height, width=width)
            _report_gl_renderer()
            self._interval = 1.0 / hz
            self._last_render_t = 0.0

    def sync_qpos(self, sim_time: float) -> None:
        """Call from the physics loop, as often as you like (cheap) -- just
        hands the render thread a fresh qpos/qvel/ctrl copy (plus the sim
        time it corresponds to) to pick up on its own schedule. Never
        blocks on rendering itself. When not threaded, just stashes
        sim_time cheaply (sync_display() reads qpos/ctrl straight off
        self.main_data instead, since it then runs on this same thread --
        no race to guard against, and no point copying first)."""
        if not self.threaded:
            self._sim_time_snapshot = sim_time
            return
        with self._qpos_lock:
            self._qpos_snapshot[:] = self.main_data.qpos
            self._qvel_snapshot[:] = self.main_data.qvel
            self._ctrl_snapshot[:] = self.main_data.ctrl
            self._sim_time_snapshot = sim_time

    def _render_loop(self, width: int, height: int, hz: float) -> None:
        own_data = mujoco.MjData(self.model)
        renderer = mujoco.Renderer(self.model, height=height, width=width)
        _report_gl_renderer()
        interval = 1.0 / hz
        try:
            while not self._stop.is_set():
                t0 = time.monotonic()
                with self._qpos_lock:
                    own_data.qpos[:] = self._qpos_snapshot
                    own_data.qvel[:] = self._qvel_snapshot
                    qpos_used = self._qpos_snapshot.copy()
                    ctrl_used = self._ctrl_snapshot.copy()
                    sim_time_used = self._sim_time_snapshot
                mujoco.mj_forward(self.model, own_data)
                images = {}
                for name in self.cam_names:
                    renderer.update_scene(own_data, camera=name)
                    images[name] = renderer.render().copy()
                # images, qpos_used and ctrl_used are all from the SAME
                # instant (the qpos/ctrl pair sync_qpos() handed over right
                # before this render pass started) -- stored together,
                # under one lock, so a reader never sees images from one
                # instant paired with joint values from another.
                with self._images_lock:
                    self.last_images.update(images)
                    self.last_qpos = qpos_used
                    self.last_ctrl = ctrl_used
                    self.last_sim_time = sim_time_used
                    self.frame_seq += 1
                elapsed = time.monotonic() - t0
                self._stop.wait(max(0.0, interval - elapsed))
        finally:
            renderer.close()

    def get_sample(self, since_seq: int = -1):
        """Returns (seq, images, qpos, ctrl, sim_time) for the most recent
        completed render batch, all mutually consistent (same instant), or
        None if seq <= since_seq (nothing new since the caller's last
        read). This is the synchronized save point for training-data
        recording: camera images paired with the EXACT joint values they
        were rendered from, not whatever's live in the physics loop right
        now (which has moved on by however long rendering took) -- that's
        what keeps images and joints on the same clock at save time even
        though rendering itself stays slow."""
        with self._images_lock:
            if self.frame_seq <= since_seq or self.last_sim_time is None:
                return None
            return (
                self.frame_seq,
                {name: img.copy() for name, img in self.last_images.items() if img is not None},
                self.last_qpos.copy(),
                self.last_ctrl.copy(),
                self.last_sim_time,
            )

    def sync_display(self) -> None:
        """Call from the physics/viewer loop as often as you like.

        threaded=True: cheap, no render() call on this path -- just pushes
        whatever the background thread has most recently rendered onto the
        matplotlib canvas (no-op if display=False).

        threaded=False: THIS is where the actual (throttled, --camera-hz)
        render happens, inline, on whichever thread calls this -- renders
        regardless of display (so --capture-cameras-only/headless mode
        still gets frames into self.last_images/get_sample()), then also
        blits to the matplotlib canvas if display=True.
        """
        if not self.threaded:
            now = time.monotonic()
            if now - self._last_render_t >= self._interval:
                self._last_render_t = now
                mujoco.mj_forward(self.model, self.main_data)  # cheap: kinematics only, qpos unchanged by this
                images = {}
                for name in self.cam_names:
                    self._renderer.update_scene(self.main_data, camera=name)
                    images[name] = self._renderer.render().copy()
                self.last_images.update(images)
                self.last_qpos = self.main_data.qpos.copy()
                self.last_ctrl = self.main_data.ctrl.copy()
                self.last_sim_time = self._sim_time_snapshot
                self.frame_seq += 1
            if not self.display:
                return
            changed = False
            for i, name in enumerate(self.cam_names):
                img = self.last_images.get(name)
                if img is not None:
                    self.images[i].set_data(img)
                    changed = True
            if changed:
                self.fig.canvas.draw_idle()
                self.fig.canvas.flush_events()
            return

        if not self.display:
            return
        with self._images_lock:
            snapshot = dict(self.last_images)
        changed = False
        for i, name in enumerate(self.cam_names):
            img = snapshot.get(name)
            if img is not None:
                self.images[i].set_data(img)
                changed = True
        if changed:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

    def close(self) -> None:
        if self.threaded:
            self._stop.set()
            self._thread.join(timeout=2.0)
        else:
            self._renderer.close()
        if self._plt is not None:
            self._plt.close(self.fig)


_SOFTWARE_GL = ("llvmpipe", "softpipe", "swrast", "gdi generic", "microsoft basic render")


def _report_gl_renderer() -> None:
    """Print which GL device the (current) camera-render context landed on,
    and warn loudly if it's a software rasterizer -- the silent failure
    mode this whole --gl-backend option exists for."""
    try:
        from OpenGL import GL
        name = GL.glGetString(GL.GL_RENDERER)
        name = name.decode() if isinstance(name, bytes) else str(name)
    except Exception as exc:  # diagnostic only, never fatal
        print(f"[mujoco_physics_process] camera GL renderer: unknown ({exc})")
        return
    backend = os.environ.get("MUJOCO_GL", "glfw (default)")
    print(f"[mujoco_physics_process] camera GL renderer: {name} [MUJOCO_GL={backend}]")
    if any(s in name.lower() for s in _SOFTWARE_GL):
        print(
            "[mujoco_physics_process] WARNING: camera rendering is on a SOFTWARE rasterizer (CPU). "
            "Linux: use --gl-backend egl. Windows laptop: set python.exe to 'High performance' "
            "in Settings > System > Display > Graphics."
        )


class TeachingDataRecorder:
    """Saves one lerobot-bound episode: per saved frame, camera images +
    joint measured (qpos) + joint commanded (ctrl), all from the SAME
    instant -- see teaching_data_generation_rule.md section 3.2, which
    requires every recorded stream to land on one shared time axis, with
    the camera's own rate as the upper bound (one image per saved frame,
    never repeated to fill a faster clock).

    Driven entirely by CameraView.get_sample()'s frame_seq: poll() only
    writes a frame when the render thread has produced a genuinely NEW
    one, so the save rate is naturally capped at whatever --camera-hz
    (and the renderer's own real throughput) actually achieves -- never
    faster, and never a duplicate image paired with fresh joint values
    that have silently drifted away from it. Rendering being slow is
    accepted as-is (see CameraView); this only guarantees that whatever
    gets saved is mutually consistent at save time.
    """

    FORMAT = "mujoco_teaching_data_v1"

    def __init__(self, out_dir: pathlib.Path, writer: MujocoJointWriter, cam_names: list, task: str):
        self.out_dir = out_dir
        self.images_dir = out_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        # Only the named robot joints, in one fixed order -- NOT a raw
        # qpos/ctrl array dump, which (with assets/scene/scene.xml loaded)
        # also holds the cloth flexcomp's own DOFs mixed in among the
        # robot's. writer._qposadr/_ctrladr give joint-name -> index for
        # exactly the robot's own joints/actuators, nothing else. Needs
        # both maps to have the name (every actuated joint does; this
        # drops anything actuator-less, there shouldn't be any on this
        # model) so observation.state and action line up index-for-index.
        self.joint_names = [n for n in writer._qposadr if n in writer._ctrladr]
        self._qpos_idx = np.array([writer._qposadr[n] for n in self.joint_names])
        self._ctrl_idx = np.array([writer._ctrladr[n] for n in self.joint_names])
        self.cam_names = list(cam_names)
        self.task = task
        self.frame_index = 0
        self._last_seq = -1
        self.file = open(out_dir / "frames.jsonl", "w", encoding="utf-8")
        json.dump({
            "type": "header", "format": self.FORMAT,
            "joint_names": self.joint_names, "camera_names": self.cam_names, "task": self.task,
        }, self.file)
        self.file.write("\n")
        self.file.flush()
        print(f"[mujoco_physics_process] recording teaching data -> {out_dir} ({len(self.joint_names)} joints)")

    def poll(self, camera_view: "CameraView") -> None:
        sample = camera_view.get_sample(self._last_seq)
        if sample is None:
            return
        seq, images, qpos, ctrl, sim_time = sample
        self._last_seq = seq
        from PIL import Image
        image_files = {}
        for name, img in images.items():
            rel = f"{name}/{self.frame_index:06d}.png"
            path = self.images_dir / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(img).save(path)
            image_files[name] = f"images/{rel}"
        record = {
            "type": "frame",
            "frame_index": self.frame_index,
            "t": sim_time,
            "observation.state": qpos[self._qpos_idx].tolist(),
            "action": ctrl[self._ctrl_idx].tolist(),
            "images": image_files,
        }
        json.dump(record, self.file)
        self.file.write("\n")
        self.file.flush()
        self.frame_index += 1

    def close(self) -> None:
        self.file.close()
        print(f"[mujoco_physics_process] recorded {self.frame_index} synced frame(s) -> {self.out_dir}")


def _start_record_console(commands: "queue.SimpleQueue[str]") -> threading.Thread:
    """Terminal start/stop control for RecordingSession, same console-
    thread pattern as the Quest calibration ones (_start_calibration_
    console / _start_quest_calibration_console) -- this process already
    keeps its main loop busy stepping physics, so recording control can't
    block on input() there; a daemon thread reading stdin and handing
    commands over through a queue is this project's established way to
    get terminal input into a real-time loop without stalling it."""
    def run() -> None:
        print(
            "[mujoco_physics_process] recording console ready: type 'record' (or 'r') then Enter "
            "to start a new episode, 'stop' (or 's') to end it and convert everything recorded so "
            "far to lerobot_ds."
        )
        for line in sys.stdin:
            cmd = line.strip().lower()
            if cmd in {"record", "r"}:
                commands.put("record")
            elif cmd in {"stop", "s"}:
                commands.put("stop")
            elif cmd:
                print("[mujoco_physics_process] unknown command; use 'record'/'r' or 'stop'/'s'")

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


class RecordingSession:
    """Terminal-controlled recording: 'record' starts a new auto-numbered
    episode (ep_000, ep_001, ... under base_dir) as a TeachingDataRecorder;
    'stop' ends it and (re)converts every ep_* episode recorded so far
    under base_dir into base_dir/lerobot_ds via scripts/convert_to_lerobot
    (the same conversion that script runs standalone -- reused here
    directly, not shelled out to, so a conversion problem surfaces
    immediately instead of silently leaving stale output). Kept as its own
    class (rather than inlining start/stop into main()'s loop) so the
    episode-numbering/conversion logic has one place to live and is
    testable without a running viewer.
    """

    def __init__(self, base_dir: pathlib.Path, writer: "MujocoJointWriter", default_task: str, fps: float):
        self.base_dir = base_dir
        self.writer = writer
        self.default_task = default_task
        self.fps = fps  # the actual achieved camera rate, i.e. args.camera_hz -- see TeachingDataRecorder
        self.recorder: Optional[TeachingDataRecorder] = None
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _next_episode_dir(self) -> pathlib.Path:
        existing = sorted(p.name for p in self.base_dir.glob("ep_*") if p.is_dir())
        next_n = 0
        for name in existing:
            try:
                next_n = max(next_n, int(name[len("ep_"):]) + 1)
            except ValueError:
                continue
        return self.base_dir / f"ep_{next_n:03d}"

    def start(self, camera_view: Optional["CameraView"]) -> None:
        if self.recorder is not None:
            print("[mujoco_physics_process] already recording an episode -- 'stop' it first")
            return
        if camera_view is None:
            print("[mujoco_physics_process] can't record -- no cameras available "
                  "(run with --show-cameras or --capture-cameras)")
            return
        ep_dir = self._next_episode_dir()
        self.recorder = TeachingDataRecorder(ep_dir, self.writer, camera_view.cam_names, self.default_task)

    def poll(self, camera_view: "CameraView") -> None:
        if self.recorder is not None:
            self.recorder.poll(camera_view)

    def stop(self) -> None:
        if self.recorder is None:
            print("[mujoco_physics_process] not currently recording")
            return
        recorder = self.recorder
        self.recorder = None
        recorder.close()
        if recorder.frame_index == 0:
            # Started then stopped before a single camera frame landed --
            # an empty episode has no frames.jsonl body at all (just the
            # header), which load_episode() rejects outright and would
            # otherwise take the WHOLE session's conversion down with it.
            # Discard it instead of leaving a landmine for the next 'stop'.
            import shutil
            shutil.rmtree(recorder.out_dir, ignore_errors=True)
            print(f"[mujoco_physics_process] {recorder.out_dir} had 0 frames, discarded")
            return
        self._convert_all()

    def _convert_all(self) -> None:
        from scripts.convert_to_lerobot import build_episode_frames, load_episode, validate_dataset, write_lerobot_dataset

        ep_dirs = sorted(p for p in self.base_dir.glob("ep_*") if p.is_dir())
        if not ep_dirs:
            print("[mujoco_physics_process] no episodes recorded yet, nothing to convert")
            return
        try:
            episodes, headers = [], []
            for ep_dir in ep_dirs:
                header, raw_frames = load_episode(ep_dir)
                episodes.append(build_episode_frames(ep_dir, header, raw_frames))
                headers.append(header)
            problems = validate_dataset(episodes, headers)
            if problems:
                print(f"[mujoco_physics_process] conversion SKIPPED -- {len(problems)} problem(s):")
                for p in problems:
                    print(f"  - {p}")
                return
            out_dir = self.base_dir / "lerobot_ds"
            write_lerobot_dataset(out_dir, episodes, headers, fps=self.fps)
            total = sum(len(e) for e in episodes)
            print(f"[mujoco_physics_process] converted {len(episodes)} episode(s), "
                  f"{total} frame(s) total -> {out_dir}")
        except Exception as exc:  # conversion failing must never crash the live viewer loop
            print(f"[mujoco_physics_process] conversion FAILED: {exc}")


class TeachingDataReplay:
    """Reads a TeachingDataRecorder episode back -- frames.jsonl + images/
    <camera>/<frame>.png, see that class -- for eyeballing a recorded
    episode against the live MuJoCo scene, the same way scripts/
    upper_body_retarget_vedo.py's CombinedReplayReader lets you eyeball a
    mocap/hand recording. Not a dataset-format shim for actual lerobot
    training -- that conversion is a separate step (see
    teaching_data_generation_rule.md)."""

    def __init__(self, path: pathlib.Path):
        self.dir = path
        lines = (path / "frames.jsonl").read_text(encoding="utf-8").splitlines()
        header = json.loads(lines[0])
        if header.get("type") != "header" or header.get("format") != TeachingDataRecorder.FORMAT:
            raise ValueError(f"{path / 'frames.jsonl'}: not a {TeachingDataRecorder.FORMAT} recording")
        self.joint_names = header["joint_names"]
        self.camera_names = header["camera_names"]
        self.task = header.get("task", "")
        self.frames = [json.loads(line) for line in lines[1:]]

    def __len__(self) -> int:
        return len(self.frames)

    def qpos_dict(self, i: int) -> dict:
        return dict(zip(self.joint_names, self.frames[i]["observation.state"]))

    def t(self, i: int) -> float:
        return self.frames[i]["t"]

    def image(self, i: int, cam_name: str):
        rel = self.frames[i]["images"].get(cam_name)
        if rel is None:
            return None
        from PIL import Image
        return np.array(Image.open(self.dir / rel))


class ReplayImageView:
    """Displays TeachingDataReplay's SAVED images (no re-rendering -- these
    are the exact frames that got recorded) next to the live MuJoCo
    viewer, same matplotlib/TkAgg window pattern as CameraView's display
    mode."""

    def __init__(self, cam_names: list):
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        self._plt = plt
        plt.ion()
        self.cam_names = cam_names
        self.fig, axes = plt.subplots(1, len(cam_names), num="Recorded cameras", figsize=(4 * len(cam_names), 3.2))
        self.axes = [axes] if len(cam_names) == 1 else list(axes)
        self.images = []
        for ax, name in zip(self.axes, cam_names):
            ax.set_title(name, fontsize=9)
            ax.axis("off")
            self.images.append(ax.imshow(np.zeros((10, 10, 3), dtype=np.uint8)))
        self.fig.tight_layout()
        self.fig.show()

    def update(self, images: dict) -> None:
        changed = False
        for i, name in enumerate(self.cam_names):
            img = images.get(name)
            if img is not None:
                self.images[i].set_data(img)
                changed = True
        if changed:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

    def close(self) -> None:
        self._plt.close(self.fig)


def _set_viewer_defaults(viewer, model: mujoco.MjModel) -> None:
    """Fixed viewpoint + hide collision-only geom groups -- shared by the
    live loop and replay, see main()'s identical block for why."""
    viewer.cam.lookat[:] = [0.0, 0.0, 1.0]
    viewer.cam.distance = 2.5
    viewer.cam.azimuth = 135
    viewer.cam.elevation = -15
    collision_only = [True, True]
    for i in range(model.ngeom):
        group = model.geom_group[i]
        if 0 <= group < len(collision_only):
            if model.geom_contype[i] == 0 and model.geom_conaffinity[i] == 0:
                collision_only[group] = False
    for group, is_collision_only in enumerate(collision_only):
        if is_collision_only:
            viewer.opt.geomgroup[group] = 0


_TARGET_MARKER_RGBA = {"left": [0.2, 1.0, 0.2, 0.85], "right": [1.0, 0.6, 0.0, 0.85]}


def _draw_wrist_target_markers(viewer, joints: dict) -> None:
    """Draws a small sphere + an oriented arrow at each arm's wrist IK
    TARGET -- sent by scripts/hand_process.py's --arm-ik quest as
    {side}_wrist_target_{x,y,z} (position) and {side}_wrist_target_r{row}
    {col} (the 3x3 target rotation matrix, see _WRIST_TARGET_NAMES/
    _WRIST_TARGET_ROT_NAMES), already in the same robot-root-fixed world
    frame the viewer renders everything else in. Lets a target-vs-achieved
    gap (the IK not fully converging -- see solve_wrist's known local-
    minimum issue) be seen directly instead of only inferred from numbers:
    the marker shows where/how the arm was TOLD to put its wrist, the
    rendered robot shows where it actually put it.

    The sphere alone can't show ORIENTATION at all (no visible features on
    a sphere, confirmed the hard way -- it looked "frozen" even while the
    target rotation was genuinely changing underneath it); the arrow is
    mjGEOM_ARROW oriented directly by that rotation matrix (`mat`), so it
    visibly spins as wrist orientation tracking does its thing.

    Only scripts/hand_process.py's --arm-ik quest sends these names, so
    this is a no-op (no markers) with any other arm source (mocap).

    viewer.user_scn is mujoco.viewer's own scratch mjvScene for exactly
    this -- geoms added here are NOT part of the compiled model and don't
    survive a reset/model change, which is fine, they're re-added fresh
    every tick from whatever receiver.joints currently holds.
    """
    viewer.user_scn.ngeom = 0
    for side, rgba in _TARGET_MARKER_RGBA.items():
        pos_keys = [f"{side}_wrist_target_{axis}" for axis in ("x", "y", "z")]
        if not all(k in joints for k in pos_keys):
            continue
        pos = np.array([joints[k] for k in pos_keys], dtype=float)

        i = viewer.user_scn.ngeom
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[i], type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.015, 0.0, 0.0], pos=pos, mat=np.eye(3).flatten(), rgba=rgba,
        )
        viewer.user_scn.ngeom += 1

        rot_keys = [f"{side}_wrist_target_r{row}{col}" for row in range(3) for col in range(3)]
        if not all(k in joints for k in rot_keys):
            continue
        mat = np.array([joints[k] for k in rot_keys], dtype=float).reshape(3, 3)
        i = viewer.user_scn.ngeom
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[i], type=mujoco.mjtGeom.mjGEOM_ARROW,
            size=[0.006, 0.006, 0.08], pos=pos, mat=mat.flatten(), rgba=rgba,
        )
        viewer.user_scn.ngeom += 1


def _run_replay(args) -> None:
    """--replay-dataset: drives qpos straight from a TeachingDataRecorder
    episode (kinematic, mj_forward-only -- this is for visual verification
    against the live scene, not re-running physics) and shows the
    recording's own saved camera images side by side, so "does this match
    what the viewer shows" is a direct visual check rather than having to
    trust the jsonl numbers. No UDP, no actuators, no recording -- those
    are the live-capture path in main()."""
    replay = TeachingDataReplay(args.replay_dataset)
    if len(replay) == 0:
        print(f"[mujoco_physics_process] {args.replay_dataset}: no frames, nothing to replay")
        return
    print(
        f"[mujoco_physics_process] replaying {len(replay)} frame(s) from {args.replay_dataset} "
        f"(task: {replay.task!r}, cameras: {replay.camera_names})"
    )

    robot_mjcf = actuated_mjcf_path(_URDF)
    model_path = robot_mjcf if args.robot_only else _SCENE
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    writer = MujocoJointWriter(model)  # replay every recorded joint, including head -- nothing skipped
    mujoco.mj_forward(model, data)

    image_view = ReplayImageView(replay.camera_names) if replay.camera_names else None

    print("[mujoco_physics_process] close the viewer window to stop")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        _set_viewer_defaults(viewer, model)
        i = 0
        prev_t = replay.t(0)
        while viewer.is_running():
            t0 = time.monotonic()
            sim_dt = max(0.0, replay.t(i) - prev_t)
            prev_t = replay.t(i)

            missing = writer.write(data, replay.qpos_dict(i))
            if missing:
                print(f"[mujoco_physics_process] WARNING: {missing} recorded joint name(s) have no "
                      "matching MuJoCo joint this frame")
            mujoco.mj_forward(model, data)
            viewer.sync()
            if image_view is not None:
                image_view.update({cam: replay.image(i, cam) for cam in replay.camera_names})

            i += 1
            if i >= len(replay):
                if not args.loop:
                    break
                i = 0
                prev_t = replay.t(0)
            wait = sim_dt / max(args.replay_speed, 1e-6) - (time.monotonic() - t0)
            if wait > 0:
                time.sleep(wait)

    if image_view is not None:
        image_view.close()


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
        "rendered via mujoco.Renderer off the same model/data as the main viewer. Implies "
        "--capture-cameras. ORs with configs/sim_params.yaml's camera.show -- set that true to "
        "turn this on by default without passing the flag every run.",
    )
    parser.add_argument(
        "--capture-cameras", action="store_true",
        help="render the cameras (CameraView.last_images) without opening a window -- for a future "
        "headless training-data-capture loop to read; no visible cost if nothing reads it. ORs with "
        "configs/sim_params.yaml's camera.capture, same as --show-cameras/camera.show.",
    )
    parser.add_argument(
        "--camera-hz", type=float, default=None,
        help="camera render rate, independent of --hz. Defaults to configs/sim_params.yaml's "
        "camera.hz if not given here. Rendering runs on CameraView's own background thread now (not "
        "the physics/viewer loop) when the GL backend allows it, so this no longer stalls physics in "
        "that case -- it just caps how current the camera images are. Kept low by default anyway "
        "since rendering is still expensive per call when it ends up on a software rasterizer "
        "(~100-1000ms for 3 cameras) and that thread can't render faster than that regardless of "
        "this setting. On a GPU context (see --gl-backend) it's ~7ms, so this can go much higher.",
    )
    parser.add_argument(
        "--no-wrist-cameras", action="store_true",
        help="with --show-cameras/--capture-cameras, render only cam_head and skip cam_left_wrist/"
        "cam_right_wrist -- cuts render cost roughly to a third since it's ~linear in camera count.",
    )
    parser.add_argument(
        "--record-dataset", type=pathlib.Path, default=None,
        help="base directory for a recording SESSION -- type 'record'/'r' then Enter in this "
        "console to start a new episode (auto-numbered ep_000, ep_001, ... under this directory), "
        "'stop'/'s' to end it. Each episode is a teaching-data recording (see "
        "teaching_data_generation_rule.md): frames.jsonl + images/<camera>/<frame>.png, one frame "
        "per NEW camera render (CameraView.frame_seq advancing), paired with the exact qpos/ctrl "
        "from that same instant -- so the save rate is capped at whatever --camera-hz actually "
        "achieves. 'stop' also (re)converts every episode recorded so far under this directory "
        "into <this directory>/lerobot_ds (see scripts/convert_to_lerobot.py). Implies "
        "--capture-cameras.",
    )
    parser.add_argument(
        "--task", type=str, default="",
        help="--record-dataset: default task description string for new episodes (saved in each "
        "episode's frames.jsonl header).",
    )
    parser.add_argument(
        "--replay-dataset", type=pathlib.Path, default=None,
        help="play back a --record-dataset episode instead of listening on --port: drives qpos "
        "straight from frames.jsonl (kinematic, no physics, no actuators) and shows the "
        "recording's own saved camera images next to the live viewer, so you can directly check "
        "a recorded episode against the same MuJoCo scene it was captured from. Ignores --port/"
        "--diagnose-latency/--show-cameras/--capture-cameras/--record-dataset.",
    )
    parser.add_argument("--replay-speed", type=float, default=1.0,
                         help="--replay-dataset: playback speed multiplier (1.0 = real recorded pace).")
    parser.add_argument("--loop", action="store_true", help="--replay-dataset: loop when the episode ends.")
    parser.add_argument(
        "--pinch-contact-force", type=float, default=None,
        help="experimental: when thumb/index distal bodies are closer than --pinch-contact-enter, "
        "maintain an artificial closing force between them. 0 disables. In the default torque "
        "mode this is converted through J^T F into qfrc_applied on the thumb/index finger joints; "
        "it is still a physics assist, not an actuator command, so it will not appear in recorded "
        "action vectors.",
    )
    parser.add_argument(
        "--pinch-contact-mode", choices=("torque", "body-force"), default=None,
        help="torque: convert fingertip force to finger-joint generalized torque via J^T F. "
        "body-force: apply the force directly to MuJoCo bodies via xfrc_applied.",
    )
    parser.add_argument(
        "--pinch-contact-enter", type=float, default=None,
        help="metres: thumb/index body distance below which --pinch-contact-force engages.",
    )
    parser.add_argument(
        "--pinch-contact-exit", type=float, default=None,
        help="metres: hysteresis release distance for --pinch-contact-force.",
    )
    _gl_backend_args(parser)
    args = parser.parse_args()

    if args.replay_dataset is not None:
        _run_replay(args)
        return

    # configs/sim_params.yaml, read fresh every run -- actuator gains,
    # cloth physics, camera resolution/rate (see robot_hand/sim_config.py).
    # force=True on actuated_mjcf_path: always rebuild from these gains,
    # never silently reuse a stale cached MJCF from a previous run/config
    # (actuated_mjcf_path's normal caching is for hand-editing the saved
    # XML directly, which this bypasses on purpose -- the YAML is the
    # source of truth here instead). generate_cloth_xml() must run BEFORE
    # scene.xml loads, since scene.xml <include>s its output.
    sim_config = load_sim_config(args.sim_config)
    generate_cloth_xml(sim_config)
    camera_hz = args.camera_hz if args.camera_hz is not None else sim_config["camera"]["hz"]
    pinch_contact_cfg = sim_config.get("pinch_contact_assist", {})
    if args.pinch_contact_force is None:
        enabled = bool(pinch_contact_cfg.get("enabled", False))
        args.pinch_contact_force = float(pinch_contact_cfg.get("force", 0.0)) if enabled else 0.0
    if args.pinch_contact_mode is None:
        args.pinch_contact_mode = str(pinch_contact_cfg.get("mode", "torque"))
    if args.pinch_contact_mode not in ("torque", "body-force"):
        raise ValueError(
            "pinch_contact_assist.mode must be 'torque' or 'body-force', "
            f"got {args.pinch_contact_mode!r}"
        )
    if args.pinch_contact_enter is None:
        args.pinch_contact_enter = float(pinch_contact_cfg.get("enter_distance", 0.025))
    if args.pinch_contact_exit is None:
        args.pinch_contact_exit = float(pinch_contact_cfg.get("exit_distance", 0.045))
    # --show-cameras/--capture-cameras OR with the config's camera.show/
    # capture -- either one passing the flag or the config saying true
    # turns it on, so the config can set "always on" without re-typing
    # the flag every run, while the flag still force-enables it for a
    # one-off run even when the config says false.
    show_cameras = args.show_cameras or bool(sim_config["camera"].get("show", False))
    capture_cameras = args.capture_cameras or bool(sim_config["camera"].get("capture", False))

    robot_mjcf = actuated_mjcf_path(_URDF, force=True, **actuator_gain_kwargs(sim_config))
    model_path = robot_mjcf if args.robot_only else _SCENE
    model = mujoco.MjModel.from_xml_path(str(model_path))
    print(f"[mujoco_physics_process] loaded {model_path}")
    print(
        "[mujoco_physics_process] pinch_contact_assist "
        f"enabled={args.pinch_contact_force > 0.0} force={args.pinch_contact_force:.2f}N "
        f"mode={args.pinch_contact_mode} enter={args.pinch_contact_enter:.3f}m "
        f"exit={args.pinch_contact_exit:.3f}m"
    )
    data = mujoco.MjData(model)
    writer = MujocoJointWriter(model, skip=_HEAD_JOINTS)
    mujoco.mj_forward(model, data)  # baseline pose for SelfCollisionMonitor's exclude set
    collision_monitor = SelfCollisionMonitor(model, data)
    receiver = ArticulationReceiver(args.port)
    pinch_force_pairs = _pinch_force_pairs(model) if args.pinch_contact_force > 0.0 else {}
    pinch_force_dofs = _pinch_force_dofs(model) if args.pinch_contact_force > 0.0 else {}
    pinch_force_active: dict[str, bool] = {}

    # Both arms extended forward instead of qpos0's limp hanging-arms pose
    # -- see _forward_reach_qpos(). Sets ctrl as well as qpos: this is a
    # position-actuated model, so leaving ctrl at its 0 default would have
    # the very first physics step immediately start servoing the arms
    # back down to 0 regardless of what qpos was just set to.
    forward_pose = _forward_reach_qpos()
    writer.write(data, forward_pose)
    writer.write_ctrl(data, forward_pose)
    mujoco.mj_forward(model, data)

    camera_view = None
    cam_names = []
    if show_cameras or capture_cameras or args.record_dataset:
        cam_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(model.ncam)
        ]
        if args.no_wrist_cameras:
            cam_names = [n for n in cam_names if n == "cam_head"]
        if not cam_names:
            print("[mujoco_physics_process] --show/capture-cameras: model has no cameras, skipping")

    recording = None
    record_commands: Optional["queue.SimpleQueue[str]"] = None
    if args.record_dataset:
        recording = RecordingSession(args.record_dataset, writer, args.task, camera_hz)
        record_commands = queue.SimpleQueue()
        _start_record_console(record_commands)

    tick_interval = 1.0 / args.hz
    substeps = max(1, round(tick_interval / model.opt.timestep))
    warned_missing = False

    print(f"[mujoco_physics_process] listening for articulation frames on UDP :{args.port}")
    print("[mujoco_physics_process] press R to reset the scene (arms-forward pose + cloth) to its initial state")
    if recording is not None:
        print("[mujoco_physics_process] press S to toggle recording start/stop (see --record-dataset)")
    print("[mujoco_physics_process] close the viewer window to stop")

    # key_callback fires on the viewer's own (background) thread, not the
    # physics-step thread below -- just flip a flag here and apply the
    # actual reset from the main loop, so it can't race a concurrent
    # mj_step. mj_resetData restores qpos to the compiled model's own
    # qpos0 (every joint at 0 -- there's no <key> keyframe), including the
    # cloth's flex vertices (ordinary generalized coordinates too, not
    # something separate) -- the arms-forward pose gets reapplied right
    # after (forward_pose, computed once above) since qpos0 itself is the
    # limp hanging-arms pose, not what reset is supposed to land on.
    reset_requested = [False]
    # Same flag-then-apply-on-main-thread pattern for S: toggles
    # recording start/stop instead of typing 'record'/'stop' into the
    # console (that still works too -- useful when the viewer window
    # isn't focused -- this is just the faster path while it is).
    record_toggle_requested = [False]

    def key_callback(keycode: int) -> None:
        if keycode in (ord("R"), ord("r")):
            reset_requested[0] = True
        elif keycode in (ord("S"), ord("s")):
            record_toggle_requested[0] = True

    try:
        with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
            _set_viewer_defaults(viewer, model)
            if cam_names:
                # Create the offscreen/aux camera renderer only after the
                # passive viewer has successfully registered its GLFW
                # window. On Windows, doing this first can leave GLFW/WGL
                # state in a bad order and make the viewer fail with
                # "Win32: Failed to register window class".
                camera_view = CameraView(
                    model, data, cam_names, display=show_cameras, hz=camera_hz,
                    width=sim_config["camera"]["width"], height=sim_config["camera"]["height"],
                )

            last_tick = 0.0
            last_diag = 0.0
            perf_ticks = 0
            perf_window_start = time.monotonic()
            while viewer.is_running():
                now = time.monotonic()
                if now - last_tick < tick_interval:
                    time.sleep(max(0.0, tick_interval - (now - last_tick)))
                    continue
                last_tick = now

                if reset_requested[0]:
                    reset_requested[0] = False
                    mujoco.mj_resetData(model, data)
                    writer.write(data, forward_pose)
                    writer.write_ctrl(data, forward_pose)
                    mujoco.mj_forward(model, data)
                    print("[mujoco_physics_process] reset to initial state (arms forward)")

                if record_toggle_requested[0]:
                    record_toggle_requested[0] = False
                    if recording is None:
                        print("[mujoco_physics_process] S pressed but no --record-dataset was given, ignoring")
                    elif recording.recorder is None:
                        recording.start(camera_view)
                    else:
                        recording.stop()

                receiver.poll()
                joint_commands = {
                    k: v for k, v in receiver.joints.items()
                    if k not in _WRIST_TARGET_NAMES and k not in _WRIST_TARGET_ROT_NAMES
                }
                missing = writer.write_ctrl(data, joint_commands)
                if missing and not warned_missing:
                    warned_missing = True
                    print(
                        f"[mujoco_physics_process] WARNING: {missing} received joint name(s) have no "
                        "matching MuJoCo actuator this tick -- printed once"
                    )
                if args.pinch_contact_force > 0.0:
                    _apply_pinch_contact_forces(
                        model,
                        data,
                        pinch_force_pairs,
                        pinch_force_dofs,
                        pinch_force_active,
                        force_n=args.pinch_contact_force,
                        enter_m=args.pinch_contact_enter,
                        exit_m=args.pinch_contact_exit,
                        mode=args.pinch_contact_mode,
                    )

                for _ in range(substeps):
                    mujoco.mj_step(model, data)
                collision_monitor.check(data, now, warn=False)
                _draw_wrist_target_markers(viewer, receiver.joints)
                viewer.sync()
                if camera_view is not None:
                    # Both cheap: sync_qpos() just hands the render thread a
                    # fresh qpos/qvel/ctrl copy (no rendering here); sync_display()
                    # just blits whatever that thread has already rendered.
                    # The actual render() cost lives entirely on CameraView's
                    # own background thread now, off this loop.
                    camera_view.sync_qpos(data.time)
                    camera_view.sync_display()
                    if recording is not None:
                        recording.poll(camera_view)

                if record_commands is not None:
                    while True:
                        try:
                            cmd = record_commands.get_nowait()
                        except queue.Empty:
                            break
                        if cmd == "record":
                            recording.start(camera_view)
                        elif cmd == "stop":
                            recording.stop()

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
                # allows, not that anything is "waiting"; there's no sleep
                # left to give back in that case.
                perf_ticks += 1
                if now - perf_window_start >= 1.0:
                    achieved_hz = perf_ticks / (now - perf_window_start)
                    print(f"[perf] target_hz={args.hz:.0f} achieved_hz={achieved_hz:.1f}")
                    perf_ticks = 0
                    perf_window_start = now
    finally:
        receiver.close()
        if camera_view is not None:
            camera_view.close()
        if recording is not None and recording.recorder is not None:
            # Viewer window closed mid-episode; finalize it (and convert
            # everything recorded this session) instead of silently dropping
            # the in-progress recording.
            recording.stop()


if __name__ == "__main__":
    main()
