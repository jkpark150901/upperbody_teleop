"""Isolated wrist-position IK test: torso fixed, no calibration/retargeting.

Drives RBY1UpperBodyRetargeter.solve_wrist() -- always the wrist link
(link_{side}_arm_6, the end-effector; shoulder/elbow are never targeted --
skips calibrate()/_targets() (the human-motion scale-free mapping + Kabsch
alignment) entirely, so this isolates one question: given a wrist target
already in the robot's own frame, how fast and how accurately does the IK
itself reach it? Useful when it's unclear whether a "wrong motion" problem
is in the IK or in the retargeting math upstream of it (see
scripts/upper_body_retarget_vedo.py for the full pipeline with real mocap
input).

Two input backends:
  --backend synthetic (default): a small orbiting target near the arm's own
    neutral wrist position -- no hardware needed.
  --backend live: the real LeftHand/RightHand raw position from MocapApi
    (same connection as mocap_hand_raw_plot.py), fed straight into
    solve_wrist as a 1:1 delta from a captured origin -- no scaling,
    rotation, or calibration in between, so anything odd is either in the
    raw sensor stream or in the IK, nothing else.

Either way, a live X/Y/Z-vs-time trace floats next to the tracked arm in
the same 3D scene (single window, no separate viewport) plotting "input"
(the raw/synthetic delta fed to the IK) against "achieved" (where the
robot's wrist actually ended up) for --chart-hand -- a jittery *input*
trace points at the sensor/stream, a smooth input with a lagging/offset
*achieved* trace points at the IK.

Usage:
    python scripts/wrist_ik_test_vedo.py --hand both
    python scripts/wrist_ik_test_vedo.py --backend live --hand left --port 7002
    python scripts/wrist_ik_test_vedo.py --hand left --iterations 3   # test fewer IK iterations
"""

from __future__ import annotations

import argparse
import ctypes
import pathlib
import sys
import time
from collections import deque
from typing import Dict, Optional

_ROOT = pathlib.Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_SDK = _ROOT / "MocapApi" / "demo" / "demo-py"
if str(_SDK) not in sys.path:
    sys.path.insert(0, str(_SDK))

import numpy as np  # noqa: E402
import vedo  # noqa: E402

from robot_hand.upper_body_retarget import RBY1UpperBodyRetargeter  # noqa: E402
from robot_hand.urdf_fk import load_urdf  # noqa: E402
from scripts.upper_body_retarget_vedo import LinkActor, _UPPER_LINKS  # noqa: E402

_URDF = _ROOT / "rby1_dg5f.urdf"
_MOCAP_HAND_NAMES = {"left": "LeftHand", "right": "RightHand"}
_CHART_COLORS = ("red3", "green4", "blue3")  # X, Y, Z


def orbit_offset(t: float, radius: float) -> np.ndarray:
    """A small closed loop, not anatomically meaningful -- just enough
    motion to see the IK track a moving target instead of a single pose."""
    return radius * np.array([np.sin(0.6 * t), 0.6 * np.cos(0.6 * t), 0.5 * np.sin(0.35 * t + 1.0)])


class LiveHandReader:
    """Raw LeftHand/RightHand global position from MocapApi -- same
    connection settings as mocap_hand_raw_plot.py, no retargeting."""

    def __init__(self, port: int):
        from MocapApi import mocap_api as mcp
        self.mcp, self.port = mcp, port
        self.app = None
        self.avatar_handle = None

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

    def poll(self, sides) -> Optional[Dict[str, np.ndarray]]:
        # Drain fully, not one poll_next_event() call per tick -- see
        # mocap_skeleton_vedo.py: a single call per tick can leave a
        # persistent backlog (Hz looks fine, every frame stays stuck ~30s
        # stale) on real hardware.
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
        missing = [_MOCAP_HAND_NAMES[s] for s in sides if _MOCAP_HAND_NAMES[s] not in joints]
        if missing:
            raise RuntimeError(f"{missing} not in mocap joints; available={sorted(joints)}")
        return {side: self._global_position(joints[_MOCAP_HAND_NAMES[side]]) for side in sides}

    def close(self):
        if self.app is not None:
            self.app.close()
            self.app = None


class WristAxes:
    """Small RGB frame (X=red, Y=green, Z=blue) at a wrist link's actual
    live pose -- position *and* orientation, unlike the plain target
    sphere which only ever carries a position (solve_wrist is position-only
    IK; the wrist's orientation is whatever the joint chain happens to
    produce, never targeted directly -- this makes that visible)."""

    def __init__(self, length: float = 0.06):
        self.length = length
        tick = np.array([0.001, 0.0, 0.0])
        self.lines = [vedo.Line([[0, 0, 0], tick], c=c, lw=5) for c in _CHART_COLORS]

    @property
    def actors(self):
        return self.lines

    def update(self, pos: np.ndarray, rot: np.ndarray):
        for axis in range(3):
            self.lines[axis].vertices = [pos, pos + rot[:, axis] * self.length]


_CHART_N_POINTS = 200  # fixed point count per trace line -- see comment in ComparisonChart


class ComparisonChart:
    """X/Y/Z-vs-time traces floating at a fixed spot in the *same* 3D scene
    (no separate viewport/window): 'input' (raw/synthetic delta fed to the
    IK) against 'achieved' (where the robot's wrist actually landed), both
    relative to the same rest position so they're directly comparable.
    Thin solid = input, thick translucent = achieved, same color per axis.
    Time runs along local +X from `origin`, value along local +Y.
    """

    def __init__(self, origin: np.ndarray, seconds: float, time_scale: float):
        self.origin, self.seconds, self.time_scale = np.asarray(origin, dtype=float), seconds, time_scale
        self.samples = deque(maxlen=6000)  # (t, in_x, in_y, in_z, ach_x, ach_y, ach_z)
        # vedo's `actor.vertices = array` only swaps the point *coordinate*
        # buffer -- it never regenerates the underlying VTK cell/topology
        # array, so a Line/Points/Lines actor never renders more primitives
        # than it had at construction (confirmed with a screenshot test:
        # growing a 1-point Points 1->19 only ever shows the original 1
        # point). Unlike mocap_skeleton_vedo.py's joint count (fixed once
        # known), this chart's sample window genuinely changes size every
        # frame, so instead each line is built ONCE at a fixed
        # _CHART_N_POINTS and draw() always resamples onto exactly that
        # many points (pad by repeating the first sample early on, decimate
        # once the window has more samples than that).
        flat = self.origin + np.linspace([0, 0, 0], [0.001, 0, 0], _CHART_N_POINTS)
        self.input_lines = [vedo.Line(flat, c=c, lw=3) for c in _CHART_COLORS]
        self.achieved_lines = [vedo.Line(flat, c=c, lw=8).alpha(0.35) for c in _CHART_COLORS]
        self.zero_ref = vedo.Line([self.origin, self.origin + [0.001, 0, 0]], c="grey5", lw=1)
        self.label = vedo.Text2D(
            "손목 델타(m) vs 시간(s) -- 얇은선=input(raw/합성), 굵은 반투명=achieved(로봇 실제) | X=빨강 Y=초록 Z=파랑",
            pos="bottom-right", s=0.85, c="black", bg="white", alpha=0.75,
        )
        self.actors = [self.zero_ref, *self.input_lines, *self.achieved_lines, self.label]

    def append(self, t: float, input_delta: np.ndarray, achieved_delta: np.ndarray):
        self.samples.append((t, *input_delta, *achieved_delta))

    def draw(self):
        if len(self.samples) < 2:
            return
        data = np.asarray(self.samples)
        data = data[data[:, 0] >= data[-1, 0] - self.seconds]
        # Resample onto exactly _CHART_N_POINTS indices -- np.interp with
        # repeated leading x-values (all early samples pinned near index 0
        # until the real window fills up) handles both the pad and the
        # decimate case through the same call.
        src_idx = np.arange(len(data))
        query_idx = np.linspace(0, len(data) - 1, _CHART_N_POINTS)
        xs = np.interp(query_idx, src_idx, (data[:, 0] - data[0, 0]) * self.time_scale)
        zeros = np.zeros(_CHART_N_POINTS)
        for axis in range(3):
            in_vals = np.interp(query_idx, src_idx, data[:, 1 + axis])
            ach_vals = np.interp(query_idx, src_idx, data[:, 4 + axis])
            self.input_lines[axis].vertices = self.origin + np.column_stack([xs, in_vals, zeros])
            self.achieved_lines[axis].vertices = self.origin + np.column_stack([xs, ach_vals, zeros])
        self.zero_ref.vertices = [self.origin + [xs[0], 0, 0], self.origin + [xs[-1], 0, 0]]


def main():
    parser = argparse.ArgumentParser(description="Isolated wrist-IK test (no calibration/retargeting)")
    parser.add_argument("--backend", choices=("synthetic", "live"), default="synthetic")
    parser.add_argument("--port", type=int, default=7002, help="MocapApi UDP port, --backend live only")
    parser.add_argument("--hand", choices=("left", "right", "both"), default="both")
    parser.add_argument("--chart-hand", choices=("left", "right"), default=None, help="default: first of --hand")
    parser.add_argument("--chart-seconds", type=float, default=5.0, help="comparison chart history window")
    parser.add_argument(
        "--chart-time-scale", type=float, default=0.1,
        help="meters per second along the chart's time axis (it's drawn in the same 3D scene, in meters)",
    )
    parser.add_argument("--radius", type=float, default=0.12, help="orbit radius, meters (--backend synthetic)")
    parser.add_argument("--speed", type=float, default=1.0, help="time multiplier for the orbit (--backend synthetic)")
    parser.add_argument("--iterations", type=int, default=7, help="damped least-squares iterations per solve")
    parser.add_argument("--calibration-pose", choices=("attention", "tpose"), default="attention")
    parser.add_argument("--hz", type=float, default=90.0, help="poll/IK tick rate")
    parser.add_argument(
        "--reset-seconds", type=float, default=3.0,
        help="countdown before the I-key start-pose reset actually happens",
    )
    parser.add_argument(
        "--render-hz", type=float, default=30.0,
        help="capped vedo redraw rate, decoupled from --hz (see upper_body_retarget_vedo.py)",
    )
    args = parser.parse_args()
    sides = ("left", "right") if args.hand == "both" else (args.hand,)
    chart_hand = args.chart_hand or sides[0]

    tree = load_urdf(str(_URDF))
    retargeter = RBY1UpperBodyRetargeter(tree, calibration_pose=args.calibration_pose)
    # No calibrate() call -- solve_wrist() bypasses _targets()/calibration
    # entirely, so self.q stays whatever set_calibration_pose() set it to
    # (the reference/rest pose) until the first solve moves the arm(s).
    base_wrist = {side: retargeter.poses()[f"link_{side}_arm_6"].pos.copy() for side in ("left", "right")}

    hand_reader = None
    if args.backend == "live":
        hand_reader = LiveHandReader(args.port)
        hand_reader.connect()

    actors = {name: LinkActor(tree.links[name], "steelblue") for name in _UPPER_LINKS}
    initial_poses = retargeter.poses()
    for name, actor in actors.items():
        actor.set_pose(initial_poses[name])

    plotter = vedo.Plotter(
        title=f"Wrist IK test ({args.backend}, torso fixed, no retargeting)", bg="white", axes=4,
    )
    plotter.add(vedo.Grid(s=(1.5, 1.5)).c("grey8").alpha(0.2))
    plotter.add([a.actor for a in actors.values()])
    markers = {side: vedo.Sphere(r=0.025, c="orange4").alpha(0.9).pos(base_wrist[side]) for side in sides}
    plotter.add(list(markers.values()))
    wrist_axes = {side: WristAxes() for side in sides}
    for wa in wrist_axes.values():
        plotter.add(wa.actors)
    stats_text = vedo.Text2D("", pos="bottom-left", s=1.0, c="black", bg="white", alpha=0.75, font="Calco")
    hint_text = vedo.Text2D(
        f"I : 시작 자세로 초기화 ({args.reset_seconds:.0f}s 후)", pos="top-right", s=1.1,
        c="white", bg="teal", alpha=0.85, font="Calco",
    )
    countdown_text = vedo.Text2D("", pos="top-left", s=1.5, c="white", bg="red3", alpha=0.9, font="Calco")
    plotter.add([stats_text, hint_text, countdown_text])

    # Chart floats in the same 3D scene next to the tracked arm -- offset
    # from its neutral wrist position so it doesn't sit inside the robot
    # geometry. Same window, same camera, rotates/pans together with the arm.
    side_sign = 1.0 if chart_hand == "left" else -1.0  # push outward, away from the body midline
    chart_origin = base_wrist[chart_hand] + np.array([0.15, side_sign * 0.35, 0.15])
    chart = ComparisonChart(chart_origin, seconds=args.chart_seconds, time_scale=args.chart_time_scale)
    plotter.add(chart.actors)
    plotter.reset_camera()  # frame the whole arm + chart, not just whatever vedo defaulted to

    render_interval = 1.0 / args.render_hz
    state = {
        "t0": time.monotonic(), "last_render": 0.0, "last_report": time.monotonic(), "frames": 0,
        "ik_ms_sum": 0.0, "err_sum_m": 0.0,
        "reset_deadline": None,
        "raw_origin": {},  # side -> np.ndarray, captured on first live frame / on reset
    }

    def start_reset_countdown():
        # 'I' now arms a countdown instead of resetting instantly -- gives
        # you time to look away from the keyboard back at the screen before
        # the arm snaps back to the start pose.
        state["reset_deadline"] = time.monotonic() + args.reset_seconds
        print(f"[wrist_ik_test] start-pose reset in {args.reset_seconds:.0f}s...")

    def do_reset_pose():
        # "시작 자세" (start pose) -- put the arm(s) back at the reference pose,
        # restart the orbit's clock, and (live backend) forget the captured
        # raw origin so it's recaptured fresh from wherever the hand is now.
        retargeter.q[:] = retargeter.reference_q.copy()
        state["t0"] = time.monotonic()
        state["reset_deadline"] = None
        state["raw_origin"] = {}
        countdown_text.text("")
        stats_text.text("시작 자세로 초기화됨").color("white").background("teal")
        print("[wrist_ik_test] reset to reference pose (start pose)")
        plotter.render()

    def update(_event=None):
        now = time.monotonic()
        if state["reset_deadline"] is not None:
            remaining = state["reset_deadline"] - now
            if remaining > 0:
                countdown_text.text(f"시작 자세로 초기화까지 {remaining:0.1f}s")
                if now - state["last_render"] >= render_interval:
                    plotter.render()
                    state["last_render"] = now
                return
            do_reset_pose()

        input_delta: Dict[str, np.ndarray] = {}
        if args.backend == "live":
            raw = hand_reader.poll(sides)
            if raw is None:
                return  # no avatar yet -- nothing to solve or chart this tick
            for side in sides:
                if side not in state["raw_origin"]:
                    state["raw_origin"][side] = raw[side].copy()
                input_delta[side] = raw[side] - state["raw_origin"][side]
        else:
            t = (time.monotonic() - state["t0"]) * args.speed
            for side in sides:
                input_delta[side] = orbit_offset(t, args.radius)

        joints = None
        ik_t0 = time.perf_counter()
        err_sum = 0.0
        for side in sides:
            target = base_wrist[side] + input_delta[side]
            joints = retargeter.solve_wrist(side, target, iterations=args.iterations)
            err_sum += retargeter.last_error_m[f"{side}_wrist"]
        state["ik_ms_sum"] += (time.perf_counter() - ik_t0) * 1000.0
        state["err_sum_m"] += err_sum / len(sides)

        # Cheap (only walks that one arm's ~9-joint chain) -- fine to run
        # every tick, unlike the full-body FK below which is render-gated.
        achieved_pos = tree.forward_kinematics(joints, only=[f"link_{chart_hand}_arm_6"])[
            f"link_{chart_hand}_arm_6"
        ].pos
        chart.append(now, input_delta[chart_hand], achieved_pos - base_wrist[chart_hand])

        if now - state["last_render"] >= render_interval:
            # joints is solve_wrist()'s return value -- the *full* current
            # joint dict (every name), not just the side last solved.
            poses = tree.forward_kinematics(joints)
            for name, actor in actors.items():
                actor.set_pose(poses[name])
            for side, marker in markers.items():
                marker.pos(base_wrist[side] + input_delta[side])
                wrist_pose = poses[f"link_{side}_arm_6"]
                wrist_axes[side].update(wrist_pose.pos, wrist_pose.as_matrix())
            chart.draw()
            plotter.render()
            state["last_render"] = now

        state["frames"] += 1
        if now - state["last_report"] >= 1.0:
            elapsed = now - state["last_report"]
            hz = state["frames"] / elapsed
            ik_ms = state["ik_ms_sum"] / state["frames"]
            err_mm = (state["err_sum_m"] / state["frames"]) * 1000.0
            line = f"poll {hz:.1f} Hz   IK {ik_ms:.2f} ms   wrist err {err_mm:.1f} mm   iters={args.iterations}"
            print(f"[wrist_ik_test] {line}")
            stats_text.text(line).color("black").background("white")
            state["last_report"], state["frames"] = now, 0
            state["ik_ms_sum"], state["err_sum_m"] = 0.0, 0.0

    def on_key(event):
        if event.keypress.lower() == "i":
            start_reset_countdown()

    # Note: plotter.add_button() is not used here -- this environment's VTK
    # build is missing vtkRenderer.AddActor2D, which vedo's Button relies
    # on, so add_button() raises AttributeError outright. The key binding
    # below plus the on-screen hint_text label are the reliable path.
    plotter.add_callback("key press", on_key)
    plotter.add_callback("timer", update)
    plotter.timer_callback("create", dt=max(1, int(1000 / args.hz)))
    print(
        f"[wrist_ik_test] backend={args.backend}, hand={args.hand}, chart_hand={chart_hand}, "
        f"iterations={args.iterations}, poll={args.hz:.0f}Hz, render(capped)={args.render_hz:.0f}Hz; "
        f"key: I = reset to start pose after a {args.reset_seconds:.0f}s countdown -- close the window to stop"
    )
    try:
        plotter.show(interactive=True)
    finally:
        if hand_reader is not None:
            hand_reader.close()
        plotter.close()


if __name__ == "__main__":
    main()
