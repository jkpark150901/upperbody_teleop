"""Raw MocapApi torso monitor: Hips + chest only, no URDF or retargeting."""

from __future__ import annotations

import argparse
import ctypes
from collections import deque
import pathlib
import sys
import time
import tkinter as tk
from tkinter import ttk

_ROOT = pathlib.Path(__file__).parent.parent
_SDK = _ROOT / "MocapApi" / "demo" / "demo-py"
if str(_SDK) not in sys.path:
    sys.path.insert(0, str(_SDK))

import numpy as np  # noqa: E402
from MocapApi import mocap_api as mcp  # noqa: E402

CHEST_CANDIDATES = ("Spine3", "Spine2", "Spine1", "Spine")
COLORS = ("#e74c3c", "#27ae60", "#3498db")


def global_xyz(joint) -> np.ndarray:
    x, y, z = ctypes.c_float(), ctypes.c_float(), ctypes.c_float()
    err = joint.api.contents.GetJointGlobalPosition(
        ctypes.pointer(x), ctypes.pointer(y), ctypes.pointer(z), joint.handle
    )
    if err != mcp.MCPError.NoError:
        raise RuntimeError(f"GetJointGlobalPosition: {mcp.MCPError._fields[err]}")
    return np.array([x.value, y.value, z.value])


class PlotPanel(ttk.Frame):
    def __init__(self, parent, title: str, seconds: float, plot_mode: tk.StringVar):
        super().__init__(parent)
        self.seconds, self.plot_mode = seconds, plot_mode
        self.samples = deque(maxlen=1200)
        self.title = ttk.Label(self, text=title, font=("Segoe UI", 13, "bold"))
        self.title.pack(anchor="w")
        self.values = ttk.Label(self, text="RPY: --     XYZ: --", font=("Consolas", 10))
        self.values.pack(anchor="w", pady=(2, 4))
        self.canvas = tk.Canvas(self, bg="#111820", height=220, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

    def append(self, now: float, rpy: np.ndarray, xyz: np.ndarray):
        self.samples.append((now, *rpy, *xyz))
        self.values.configure(
            text=(f"RAW RPY° [{rpy[0]:7.2f}, {rpy[1]:7.2f}, {rpy[2]:7.2f}]    "
                  f"GLOBAL XYZ [{xyz[0]:8.4f}, {xyz[1]:8.4f}, {xyz[2]:8.4f}]")
        )

    def clear(self):
        self.samples.clear()

    def draw(self):
        c = self.canvas
        c.delete("all")
        w, h = max(c.winfo_width(), 20), max(c.winfo_height(), 20)
        ml, mt, mr, mb = 48, 10, 10, 24
        c.create_rectangle(ml, mt, w - mr, h - mb, outline="#52606d")
        if len(self.samples) < 2:
            c.create_text(w / 2, h / 2, text="RAW 데이터 대기 중", fill="#c7d0d9")
            return
        data = np.asarray(self.samples)
        data = data[data[:, 0] >= data[-1, 0] - self.seconds]
        values = data[:, 1:4] if self.plot_mode.get() == "rotation" else data[:, 4:7]
        lo, hi = float(values.min()), float(values.max())
        pad = max((hi - lo) * 0.12, 1.0)
        lo, hi = lo - pad, hi + pad
        for i in range(5):
            y = mt + i * (h - mt - mb) / 4
            v = hi - i * (hi - lo) / 4
            c.create_line(ml, y, w - mr, y, fill="#263442")
            c.create_text(ml - 5, y, text=f"{v:.1f}", fill="#9aa8b5", anchor="e")
        t0, t1 = data[0, 0], max(data[-1, 0], data[0, 0] + 1e-6)
        for axis, color in enumerate(COLORS):
            points = []
            for sample_index, row in enumerate(data):
                x = ml + (row[0] - t0) / (t1 - t0) * (w - ml - mr)
                y = mt + (hi - values[sample_index, axis]) / (hi - lo) * (h - mt - mb)
                points.extend((x, y))
            if len(points) >= 4:
                c.create_line(*points, fill=color, width=2)


class RawTorsoMonitor:
    def __init__(self, root: tk.Tk, port: int, seconds: float):
        self.root, self.port = root, port
        self.app = None
        self.avatar_handle = None
        self.chest_name = None
        self.frames = 0
        self.report_time = time.monotonic()
        self.last_arrival = None
        self.arrival_dts = deque(maxlen=120)
        self.max_batch = 0
        self.status = tk.StringVar(value=f"UDP {port} 수신 대기")
        self.plot_mode = tk.StringVar(value="rotation")

        root.title("Mocap RAW — Hips + Chest")
        root.geometry("1100x680")
        root.minsize(780, 520)
        root.protocol("WM_DELETE_WINDOW", self.close)
        outer = ttk.Frame(root, padding=10)
        outer.pack(fill="both", expand=True)
        bar = ttk.Frame(outer)
        bar.pack(fill="x", pady=(0, 8))
        ttk.Label(bar, text="MocapApi RAW (리타게팅 없음)", font=("Segoe UI", 15, "bold")).pack(side="left")
        ttk.Button(bar, text="그래프 초기화", command=self.clear).pack(side="right")
        ttk.Label(outer, textvariable=self.status, font=("Consolas", 10)).pack(anchor="w", pady=(0, 8))
        controls = ttk.Frame(outer)
        controls.pack(fill="x", pady=(0, 6))
        ttk.Label(controls, text="그래프:").pack(side="left")
        ttk.Radiobutton(controls, text="RAW 로컬 회전", variable=self.plot_mode, value="rotation").pack(side="left")
        ttk.Radiobutton(controls, text="RAW 글로벌 위치", variable=self.plot_mode, value="position").pack(side="left")
        ttk.Label(controls, text="  X=빨강   Y=초록   Z=파랑").pack(side="left")
        self.hips = PlotPanel(outer, "Hips (허리/골반)", seconds, self.plot_mode)
        self.hips.pack(fill="both", expand=True, pady=(0, 8))
        self.chest = PlotPanel(outer, "Chest (검색 중)", seconds, self.plot_mode)
        self.chest.pack(fill="both", expand=True)
        self.connect()
        root.after(1, self.update)

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
        # Cache mode preserves every event and can make a diagnostic viewer
        # visibly trail behind. For teleoperation we want the freshest frame.
        self.app.disable_event_cache()

    def clear(self):
        self.hips.clear()
        self.chest.clear()

    def update(self):
        try:
            newest = False
            events = self.app.poll_next_event()
            avatar_events = 0
            for event in events:
                if event.event_type == mcp.MCPEventType.AvatarUpdated:
                    self.avatar_handle = event.event_data.avatar_handle
                    newest = True
                    avatar_events += 1
            if newest:
                self.max_batch = max(self.max_batch, avatar_events)
                now = time.monotonic()
                if self.last_arrival is not None:
                    self.arrival_dts.append(now - self.last_arrival)
                self.last_arrival = now
                avatar = mcp.MCPAvatar(self.avatar_handle)
                joints = {joint.get_name(): joint for joint in avatar.get_joints()}
                if "Hips" not in joints:
                    raise RuntimeError(f"Hips 없음; joints={sorted(joints)}")
                if self.chest_name not in joints:
                    self.chest_name = next((n for n in CHEST_CANDIDATES if n in joints), None)
                    if self.chest_name is None:
                        raise RuntimeError(f"Spine 계열 없음; joints={sorted(joints)}")
                    self.chest.title.configure(text=f"Chest = {self.chest_name} (가슴)")
                for panel, name in ((self.hips, "Hips"), (self.chest, self.chest_name)):
                    joint = joints[name]
                    # Direct SDK values. No coordinate conversion, filtering, calibration, IK, or URDF.
                    rpy = np.asarray(joint.get_local_rotation_by_euler(), dtype=float)
                    panel.append(now, rpy, global_xyz(joint))
                self.frames += 1
                if now - self.report_time >= 1.0:
                    elapsed = now - self.report_time
                    hz = self.frames / elapsed
                    mean_ms = 1000 * float(np.mean(self.arrival_dts)) if self.arrival_dts else 0.0
                    jitter_ms = 1000 * float(np.std(self.arrival_dts)) if self.arrival_dts else 0.0
                    self.status.set(
                        f"UDP {self.port} ● RAW 수신 {hz:5.1f} Hz | "
                        f"frame interval {mean_ms:6.2f} ms | jitter {jitter_ms:5.2f} ms | "
                        f"batch max {self.max_batch} | chest={self.chest_name}"
                    )
                    self.frames, self.report_time, self.max_batch = 0, now, 0
                self.hips.draw()
                self.chest.draw()
        except Exception as exc:
            self.status.set(f"오류: {exc}")
        finally:
            if self.app is not None:
                self.root.after(1, self.update)

    def close(self):
        app, self.app = self.app, None
        if app is not None:
            app.close()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="Raw Hips/Chest MocapApi GUI (no retargeting)")
    parser.add_argument("--port", type=int, default=7002)
    parser.add_argument("--seconds", type=float, default=8.0)
    args = parser.parse_args()
    root = tk.Tk()
    RawTorsoMonitor(root, args.port, args.seconds)
    root.mainloop()


if __name__ == "__main__":
    main()
