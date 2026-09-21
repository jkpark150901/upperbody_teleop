"""Full-body MocapApi RAW skeleton viewer -- no retargeting/URDF.

Lists every joint currently present on the live BVH stream (console, once,
as soon as the first frame arrives) and draws the whole skeleton live in 3D:
every joint as a point, every parent-child link as a bone. Same "just show
me the raw stream" spirit as mocap_hand_raw_plot.py, but whole-body instead
of hands-only, since which joints exist and how they're named can vary by
Axis Studio actor/template -- this is also the quickest way to find out.

Can also record whatever it's showing to a file (--record) and play a
recording back later with no hardware attached (--backend replay), timed to
match the original capture's own pacing -- handy for a repeatable demo, or
for re-running the same real motion through downstream tools (e.g. feeding
a recorded trajectory into wrist_ik_test_vedo.py-style testing) without a
live suit every time.

Usage:
    python scripts/mocap_skeleton_vedo.py --port 7002
    python scripts/mocap_skeleton_vedo.py --backend mock                      # no hardware needed
    python scripts/mocap_skeleton_vedo.py --port 7002 --record demo.jsonl     # live + record
    python scripts/mocap_skeleton_vedo.py --backend replay --replay-file demo.jsonl
"""

from __future__ import annotations

import argparse
import bisect
import ctypes
import json
import math
import pathlib
import sys
import time
from typing import Dict, List, Optional, Tuple

_ROOT = pathlib.Path(__file__).parent.parent
_SDK = _ROOT / "MocapApi" / "demo" / "demo-py"
if str(_SDK) not in sys.path:
    sys.path.insert(0, str(_SDK))

import numpy as np  # noqa: E402
import vedo  # noqa: E402
from MocapApi import mocap_api as mcp  # noqa: E402


def global_xyz(joint) -> np.ndarray:
    x, y, z = ctypes.c_float(), ctypes.c_float(), ctypes.c_float()
    err = joint.api.contents.GetJointGlobalPosition(
        ctypes.pointer(x), ctypes.pointer(y), ctypes.pointer(z), joint.handle
    )
    if err != mcp.MCPError.NoError:
        raise RuntimeError(f"GetJointGlobalPosition: {mcp.MCPError._fields[err]}")
    return np.array([x.value, y.value, z.value], dtype=float)


def build_hierarchy(avatar) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Walk the live joint tree once via get_children() (not the tag-based
    static API, which needs the SDK's skeleton-template tags rather than
    whatever this particular actor is actually sending) -- returns the
    joint names in tree order plus (parent_name, child_name) bone edges."""
    names: List[str] = []
    edges: List[Tuple[str, str]] = []

    def visit(joint):
        name = joint.get_name()
        names.append(name)
        for child in joint.get_children():
            edges.append((name, child.get_name()))
            visit(child)

    visit(avatar.get_root_joint())
    return names, edges


class SkeletonMonitor:
    def __init__(self, port: int):
        self.port = port
        self.app = None
        self.avatar_handle = None
        self.names: Optional[List[str]] = None
        self.edges: Optional[List[Tuple[str, str]]] = None
        self.positions: Dict[str, np.ndarray] = {}
        self.frames = 0
        self.report_time = time.monotonic()
        self.last_latency_ms: Optional[float] = None
        self._posture_time_supported = True
        self._posture_index_supported = True
        self.last_posture_index: Optional[int] = None

    def connect(self):
        settings = mcp.MCPSettings()
        settings.set_udp(self.port)
        settings.set_bvh_data(mcp.MCPBvhData.Binary)
        settings.set_bvh_transformation(mcp.MCPBvhDisplacement.Enable)
        settings.set_bvh_rotation(mcp.MCPBvhRotation.YXZ)
        self.app = mcp.MCPApplication()
        self.app.set_settings(settings)
        ok, message = self.app.open()
        if not ok:
            raise RuntimeError(f"UDP {self.port} open failed: {message}")
        # Freshest frame only -- see mocap_hand_raw_plot.py for why.
        self.app.disable_event_cache()

    def _capture_latency_ms(self, avatar) -> Optional[float]:
        """Local wall clock minus this posture's device-side timestamp --
        same diagnostic as upper_body_retarget_vedo.py's
        MocapLandmarkReader._capture_latency_ms: a Hz counter can look
        perfectly healthy while every frame it's showing is still, say,
        300ms stale, because Hz only measures how often frames arrive, not
        how old each one already was when it did. Only meaningful when
        this process and the mocap PC share a clock (same machine, or
        NTP-synced).
        """
        if not self._posture_time_supported:
            return None
        try:
            hour, minute, second, millisecond = avatar.get_avatar_posture_time()
        except Exception:
            self._posture_time_supported = False
            print(
                "[mocap_skeleton] get_avatar_posture_time() unsupported on this MocapApi "
                "stream (capture latency will show N/A; data/loop Hz are unaffected)"
            )
            return None
        device_ms = ((hour * 60 + minute) * 60 + second) * 1000 + millisecond
        local = time.localtime()
        local_ms = ((local.tm_hour * 60 + local.tm_min) * 60 + local.tm_sec) * 1000
        local_ms += int((time.time() % 1) * 1000)
        latency_ms = local_ms - device_ms
        if latency_ms < -12 * 3600 * 1000:
            latency_ms += 24 * 3600 * 1000
        elif latency_ms > 12 * 3600 * 1000:
            latency_ms -= 24 * 3600 * 1000
        return float(latency_ms)

    def poll(self) -> bool:
        """Pull the latest avatar frame into self.positions.

        Cheap on purpose (no vedo/rendering calls) so it can run every tick
        at the mocap's own rate -- rendering is driven by a separate, capped
        timer in main(), same fix as mocap_hand_raw_plot.py/
        upper_body_retarget_vedo.py's poll-vs-render split.
        """
        newest = False
        # Drain fully, not just one poll_next_event() call: the wrapper's
        # poll_next_event() itself asks the native side "how many are
        # pending" then fetches exactly that many in one shot, so this loop
        # is normally a no-op after the first iteration -- but if
        # disable_event_cache() (below) isn't actually preventing backlog
        # for some stream/SDK-build combination, a single call per tick
        # would only ever chip away at a growing backlog at the same rate
        # it refills, silently pinning the displayed pose however far
        # behind it started (looks like healthy Hz the whole time, because
        # Hz only measures how often *a* frame arrives, not how stale it
        # is). Looping here guarantees we always end each poll() at the
        # true latest, however large the backlog.
        for _ in range(64):
            events = self.app.poll_next_event()
            if not events:
                break
            for event in events:
                if event.event_type == mcp.MCPEventType.AvatarUpdated:
                    self.avatar_handle = event.event_data.avatar_handle
                    newest = True
        if not newest or self.avatar_handle is None:
            return False
        avatar = mcp.MCPAvatar(self.avatar_handle)
        self.last_latency_ms = self._capture_latency_ms(avatar)
        if self._posture_index_supported:
            try:
                self.last_posture_index = avatar.get_avatar_posture_index()
            except Exception:
                self._posture_index_supported = False
                self.last_posture_index = None
        if self.names is None:
            self.names, self.edges = build_hierarchy(avatar)
            print(f"[mocap_skeleton] {len(self.names)} joints on this stream (root first):")
            for name in self.names:
                print(f"  - {name}")
        for joint in avatar.get_joints():
            self.positions[joint.get_name()] = global_xyz(joint)
        self.frames += 1
        return True

    def close(self):
        if self.app is not None:
            self.app.close()
            self.app = None


# Parent-relative rest offsets (meters, Y-up to match Axis Neuron's BVH
# convention elsewhere in this repo) for a plain humanoid stick figure --
# not a real skeleton, just enough bones to exercise build_hierarchy's
# analog and the render path with no hardware attached.
_MOCK_BONES: List[Tuple[str, str, Tuple[float, float, float]]] = [
    ("Hips", "Spine", (0.0, 0.15, 0.0)),
    ("Spine", "Spine1", (0.0, 0.15, 0.0)),
    ("Spine1", "Neck", (0.0, 0.12, 0.0)),
    ("Neck", "Head", (0.0, 0.12, 0.0)),
    ("Spine1", "LeftShoulder", (0.08, 0.05, 0.0)),
    ("LeftShoulder", "LeftArm", (0.15, 0.0, 0.0)),
    ("LeftArm", "LeftForeArm", (0.28, 0.0, 0.0)),
    ("LeftForeArm", "LeftHand", (0.25, 0.0, 0.0)),
    ("Spine1", "RightShoulder", (-0.08, 0.05, 0.0)),
    ("RightShoulder", "RightArm", (-0.15, 0.0, 0.0)),
    ("RightArm", "RightForeArm", (-0.28, 0.0, 0.0)),
    ("RightForeArm", "RightHand", (-0.25, 0.0, 0.0)),
    ("Hips", "LeftUpLeg", (0.09, -0.05, 0.0)),
    ("LeftUpLeg", "LeftLeg", (0.0, -0.45, 0.0)),
    ("LeftLeg", "LeftFoot", (0.0, -0.42, 0.05)),
    ("Hips", "RightUpLeg", (-0.09, -0.05, 0.0)),
    ("RightUpLeg", "RightLeg", (0.0, -0.45, 0.0)),
    ("RightLeg", "RightFoot", (0.0, -0.42, 0.05)),
]


class MockSkeletonSource:
    """Synthetic animated skeleton -- same public shape as SkeletonMonitor
    (names/edges/positions/frames, connect/poll/close) so main()'s render
    loop doesn't need to know which one it's driving."""

    def __init__(self):
        self.names = ["Hips"] + [child for _, child, _ in _MOCK_BONES]
        self.edges = [(parent, child) for parent, child, _ in _MOCK_BONES]
        self.positions: Dict[str, np.ndarray] = {}
        self.frames = 0
        self.t0 = time.monotonic()

    def connect(self):
        self.t0 = time.monotonic()
        print(f"[mocap_skeleton] mock backend: {len(self.names)} synthetic joints (root first):")
        for name in self.names:
            print(f"  - {name}")

    def _offset(self, parent: str, child: str, base: Tuple[float, float, float], t: float) -> np.ndarray:
        dx, dy, dz = base
        # A few bones sway over time so the mock skeleton visibly "moves"
        # instead of sitting in one static rest pose.
        if child in ("LeftArm", "RightArm"):
            dz += 0.18 * math.sin(0.8 * t + (0.0 if child == "LeftArm" else math.pi))
        elif child in ("LeftForeArm", "RightForeArm"):
            dz += 0.12 * math.sin(0.9 * t + 1.0 + (0.0 if child == "LeftForeArm" else math.pi))
        elif child == "Head":
            dx += 0.05 * math.sin(0.5 * t)
        elif child in ("LeftUpLeg", "RightUpLeg"):
            dz += 0.05 * math.sin(0.6 * t + (0.0 if child == "LeftUpLeg" else math.pi))
        return np.array([dx, dy, dz])

    def poll(self) -> bool:
        t = time.monotonic() - self.t0
        self.positions["Hips"] = np.array([0.0, 1.0 + 0.02 * math.sin(0.3 * t), 0.0])
        for parent, child, base in _MOCK_BONES:
            self.positions[child] = self.positions[parent] + self._offset(parent, child, base, t)
        self.frames += 1
        return True

    def close(self):
        pass


class Recorder:
    """Appends whatever a live/mock source produces to a JSON-lines file:
    one header line ({"type":"header","names":[...],"edges":[[p,c],...]})
    written once topology is known, then one {"type":"frame","t":...,
    "pos":{...}} line per polled frame, `t` relative to when recording
    started. Plain JSON lines (not a binary format) so it's easy to
    inspect/diff/hand-edit, and ReplaySkeletonSource below reads exactly
    this format back.
    """

    def __init__(self, path: pathlib.Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(path, "w", encoding="utf-8")
        self.t0 = time.monotonic()
        self.header_written = False
        self.frames_written = 0

    def write_header(self, names: List[str], edges: List[Tuple[str, str]]):
        if self.header_written:
            return
        json.dump({"type": "header", "names": names, "edges": [list(e) for e in edges]}, self.file)
        self.file.write("\n")
        self.file.flush()  # get this safely on disk before any frame writes --
        # an ungraceful shutdown (killed process, native crash) mid-recording
        # otherwise risks the OS never having flushed it at all, or flushing a
        # torn/partial header line. Frame lines don't need this: losing the
        # last one or two to an abrupt kill is harmless, but losing the only
        # topology info in the file is not (see ReplaySkeletonSource's
        # fallback for recordings made before this fix).
        self.header_written = True

    def write_frame(self, positions: Dict[str, np.ndarray]):
        t = time.monotonic() - self.t0
        pos = {name: [round(float(v), 5) for v in xyz] for name, xyz in positions.items()}
        json.dump({"type": "frame", "t": round(t, 4), "pos": pos}, self.file)
        self.file.write("\n")
        self.frames_written += 1

    def close(self):
        self.file.close()
        print(f"[mocap_skeleton] recorded {self.frames_written} frames -> {self.path}")


class ReplaySkeletonSource:
    """Plays back a Recorder file -- same public shape as SkeletonMonitor/
    MockSkeletonSource (names/edges/positions/frames, connect/poll/close)
    so main()'s render loop doesn't need to know it's not live. Paced by
    the recording's own relative timestamps (not the poll rate this is
    called at), so a recording made at ~90Hz still plays back at ~90Hz
    worth of real motion regardless of how often poll() is invoked.
    """

    def __init__(self, path: pathlib.Path, loop: bool = True):
        self.loop = loop
        self.names: Optional[List[str]] = None
        self.edges: Optional[List[Tuple[str, str]]] = None
        self.positions: Dict[str, np.ndarray] = {}
        self.frames = 0
        records: List[Tuple[float, Dict[str, np.ndarray]]] = []
        all_keys: set = set()
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
                    # torn/partial line (seen in practice: the header line
                    # itself, cut off mid-write). Skip it and keep going --
                    # losing one line out of a thousand shouldn't sink the
                    # whole recording.
                    bad_lines += 1
                    continue
                if obj.get("type") == "header":
                    self.names = obj["names"]
                    self.edges = [tuple(e) for e in obj["edges"]]
                elif obj.get("type") == "frame":
                    pos = {k: np.array(v, dtype=float) for k, v in obj["pos"].items()}
                    records.append((obj["t"], pos))
                    all_keys.update(pos.keys())
        if bad_lines:
            print(f"[mocap_skeleton] WARNING: skipped {bad_lines} corrupt/unparseable line(s) in {path}")
        if not records:
            raise RuntimeError(f"No usable frames in recording: {path}")
        if self.names is None:
            # Header line missing or corrupt (its topology info -- edges --
            # is gone for good), but every frame still carries its own full
            # set of joint names in "pos", so positions can still be
            # recovered and played back; just no bone lines to draw.
            self.names = sorted(all_keys)
            self.edges = []
            print(
                f"[mocap_skeleton] WARNING: no usable header in {path} -- recovered "
                f"{len(self.names)} joint names from frame data instead; bones won't be drawn"
            )
        self._records = records
        self._times = [t for t, _ in records]
        self._duration = self._times[-1]
        self.t0: Optional[float] = None
        print(f"[mocap_skeleton] loaded recording: {len(records)} frames, {self._duration:.1f}s, {len(self.names)} joints ({path})")

    def connect(self):
        self.t0 = time.monotonic()

    def poll(self) -> bool:
        if self.t0 is None:
            self.connect()
        elapsed = time.monotonic() - self.t0
        if self._duration <= 0:
            t = 0.0
        elif self.loop:
            t = elapsed % self._duration
        else:
            t = min(elapsed, self._duration)
        # Latest recorded frame at or before t -- stateless per call (no
        # running index to get confused by the loop wrapping back to t=0).
        idx = max(0, min(bisect.bisect_right(self._times, t) - 1, len(self._records) - 1))
        self.positions = self._records[idx][1]
        self.frames += 1
        return True

    def close(self):
        pass


def main():
    parser = argparse.ArgumentParser(description="Raw full-body MocapApi skeleton viewer (no retargeting)")
    parser.add_argument("--backend", choices=("live", "mock", "replay"), default="live", help="mock/replay need no hardware")
    parser.add_argument("--port", type=int, default=7002)
    parser.add_argument(
        "--render-hz", type=float, default=30.0,
        help="capped vedo redraw rate, decoupled from the mocap poll rate",
    )
    parser.add_argument(
        "--record", type=pathlib.Path, default=None,
        help="also write every frame to this .jsonl file as it streams (works with --backend live or mock)",
    )
    parser.add_argument("--replay-file", type=pathlib.Path, default=None, help="--backend replay: a --record'ed .jsonl file")
    parser.add_argument("--no-loop", action="store_true", help="--backend replay: play once and hold on the last frame instead of looping")
    args = parser.parse_args()
    if args.backend == "replay" and args.replay_file is None:
        parser.error("--backend replay requires --replay-file")

    if args.backend == "live":
        monitor = SkeletonMonitor(args.port)
    elif args.backend == "mock":
        monitor = MockSkeletonSource()
    else:
        monitor = ReplaySkeletonSource(args.replay_file, loop=not args.no_loop)
    monitor.connect()

    recorder = Recorder(args.record) if args.record is not None else None

    plotter = vedo.Plotter(title=f"MocapApi RAW skeleton ({args.backend})", bg="white", axes=4)
    plotter.add(vedo.Grid(s=(2, 2)).c("grey8").alpha(0.2))
    status_text = vedo.Text2D("", pos="top-left", s=1.1, c="black", bg="white", alpha=0.8, font="Calco")
    plotter.add(status_text)
    if recorder is not None:
        rec_text = vedo.Text2D("REC", pos="top-right", s=1.4, c="white", bg="red3", alpha=0.9, font="Calco")
        plotter.add(rec_text)

    render_interval = 1.0 / args.render_hz
    state = {
        "camera_fit": False, "last_render": 0.0,
        "last_report": time.monotonic(), "frames": 0, "ticks": 0,
        # Built lazily once joint/edge counts are known -- see the note in
        # update() below on why they can't start as small placeholders.
        "joints_actor": None, "bones_actor": None,
        "render_ms_sum": 0.0, "render_n": 0,
        "latency_sum_ms": 0.0, "latency_n": 0,
        "idx_at_report_start": None,
    }

    def update(_event=None):
        # One timer, internally gated -- vedo/VTK delivers every "timer"
        # callback on every TimerEvent regardless of which dt created it
        # (there's no per-callback timer routing), so running poll/render/
        # report as three separately-scheduled timers would fire all three
        # at whichever is the fastest dt. Same poll-vs-render split as
        # mocap_hand_raw_plot.py, just done with one callback instead.
        polled = monitor.poll()
        now = time.monotonic()
        state["ticks"] += 1

        if recorder is not None and polled and monitor.names and monitor.positions:
            # Gate on `polled`, not just "positions exist" -- otherwise a
            # stall (no new avatar frame) would still write the same stale
            # position repeatedly, inflating the frame count and quietly
            # lying about the recording's actual temporal resolution.
            recorder.write_header(monitor.names, monitor.edges)
            recorder.write_frame(monitor.positions)

        if now - state["last_render"] >= render_interval:
            if monitor.names and monitor.positions:
                pts = np.array([monitor.positions.get(n, (0.0, 0.0, 0.0)) for n in monitor.names])
                if state["joints_actor"] is None:
                    # vedo's `actor.vertices = bigger_array` only replaces the
                    # point *coordinate* buffer -- it never regenerates the
                    # underlying VTK cell/topology array, so a Points/Lines
                    # actor never renders more primitives than it had at
                    # construction (confirmed: growing a 1-point Points from
                    # 1->19 shows only the original 1 point on screen, even
                    # though .vertices correctly reports 19 rows). Build at
                    # the real, final joint count once instead of growing a
                    # placeholder -- joint count is fixed for the life of a
                    # run, so every later frame is a same-size update, which
                    # *does* render correctly.
                    state["joints_actor"] = vedo.Points(pts, r=10, c="red4")
                    plotter.add(state["joints_actor"])
                else:
                    state["joints_actor"].vertices = pts
                if monitor.edges:
                    starts = np.array([monitor.positions.get(p, (0.0, 0.0, 0.0)) for p, _ in monitor.edges])
                    ends = np.array([monitor.positions.get(c, (0.0, 0.0, 0.0)) for _, c in monitor.edges])
                    if state["bones_actor"] is None:
                        state["bones_actor"] = vedo.Lines(starts, ends, lw=3, c="steelblue")
                        plotter.add(state["bones_actor"])
                    else:
                        interleaved = np.empty((2 * len(monitor.edges), 3))
                        interleaved[0::2] = starts
                        interleaved[1::2] = ends
                        state["bones_actor"].vertices = interleaved
                if not state["camera_fit"]:
                    plotter.reset_camera()
                    state["camera_fit"] = True
            # Timed separately so a "looks laggy" complaint can be pinned on
            # this specific VTK draw call rather than guessed at -- a slow
            # or jittery render can make things feel behind even when data
            # Hz above is perfectly healthy.
            render_t0 = time.perf_counter()
            plotter.render()
            state["render_ms_sum"] += (time.perf_counter() - render_t0) * 1000.0
            state["render_n"] += 1
            state["last_render"] = now

        if polled:
            state["frames"] += 1
            latency_ms = getattr(monitor, "last_latency_ms", None)
            if latency_ms is not None:
                state["latency_sum_ms"] += latency_ms
                state["latency_n"] += 1
        if now - state["last_report"] >= 1.0:
            elapsed = now - state["last_report"]
            # `data` Hz = actual new avatar frames (what you want to know
            # when asking "is this real-time") vs `loop` Hz = how often
            # this timer callback itself fires, which used to be reported
            # as "poll Hz" even on ticks with no new data -- misleadingly
            # high (matching the timer's ~1ms schedule) regardless of
            # whether the mocap stream was actually keeping up.
            data_hz = state["frames"] / elapsed
            loop_hz = state["ticks"] / elapsed
            render_ms = state["render_ms_sum"] / state["render_n"] if state["render_n"] else 0.0
            if state["latency_n"] > 0:
                latency_label = f"{state['latency_sum_ms'] / state['latency_n']:.0f}ms"
            else:
                latency_label = "N/A"  # mock/replay backend, or unsupported get_avatar_posture_time()
            n_joints = len(monitor.names) if monitor.names else 0
            rec_suffix = f"   [REC {recorder.frames_written}f]" if recorder is not None else ""
            # Native posture-index delta over this ~1s window, straight from
            # the SDK's own frame counter -- if this keeps climbing well
            # past `data` Hz report after report (rather than settling to a
            # stable ratio), the backlog is growing inside MocapApi itself,
            # independent of anything in this script's own loop/render.
            idx_now = getattr(monitor, "last_posture_index", None)
            if idx_now is not None and state["idx_at_report_start"] is not None:
                idx_label = f"{(idx_now - state['idx_at_report_start']) / elapsed:.1f}/s"
            else:
                idx_label = "N/A"
            state["idx_at_report_start"] = idx_now
            print(
                f"[mocap_skeleton] data={data_hz:.1f}Hz loop={loop_hz:.0f}Hz "
                f"render={render_ms:.1f}ms capture->display={latency_label} posture_idx_rate={idx_label}"
            )
            status_text.text(
                f"joints={n_joints}   data={data_hz:.1f} Hz   loop={loop_hz:.0f} Hz   "
                f"render={render_ms:.1f} ms   latency={latency_label}{rec_suffix}"
                if n_joints else "joint 스트림 대기 중..."
            )
            state["last_report"], state["frames"], state["ticks"] = now, 0, 0
            state["render_ms_sum"], state["render_n"] = 0.0, 0
            state["latency_sum_ms"], state["latency_n"] = 0.0, 0

    plotter.add_callback("timer", update)
    plotter.timer_callback("create", dt=1)

    source_label = {
        "live": f"UDP={args.port}", "mock": "mock (synthetic)", "replay": f"replay={args.replay_file}",
    }[args.backend]
    record_note = f", recording -> {args.record}" if recorder is not None else ""
    print(
        f"[mocap_skeleton] backend={args.backend}, {source_label}, "
        f"render(capped)={args.render_hz:.0f}Hz{record_note} -- close the window to stop"
    )
    try:
        plotter.show(interactive=True)
    finally:
        monitor.close()
        if recorder is not None:
            recorder.close()
        plotter.close()


if __name__ == "__main__":
    main()
