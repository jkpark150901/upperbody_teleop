"""Small live GUI for identifying Axis Neuron joints and inspecting XYZ motion.

Usage:
    python scripts/mocap_joint_plot.py --port 7002
"""

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
for path in (_ROOT, _SDK):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np  # noqa: E402
from MocapApi import mocap_api as mcp  # noqa: E402


COLORS = ("#e74c3c", "#27ae60", "#2980b9")
LABELS = ("X", "Y", "Z")


def global_position(joint) -> np.ndarray:
    x, y, z = ctypes.c_float(), ctypes.c_float(), ctypes.c_float()
    err = joint.api.contents.GetJointGlobalPosition(
        ctypes.pointer(x), ctypes.pointer(y), ctypes.pointer(z), joint.handle
    )
    if err != mcp.MCPError.NoError:
        raise RuntimeError(f"GetJointGlobalPosition: {mcp.MCPError._fields[err]}")
    return np.array([x.value, y.value, z.value])


class MocapJointPlot:
    def __init__(self, root: tk.Tk, port: int, seconds: float = 8.0):
        self.root, self.port, self.seconds = root, port, seconds
        self.app = None
        self.avatar_handle = None
        self.joints = {}
        self.samples = deque(maxlen=600)
        self.last_frame = time.monotonic()
        self.frame_count = 0
        self.selected = tk.StringVar(value="")
        self.status = tk.StringVar(value=f"UDP {port} 연결 중...")
        self.value_text = tk.StringVar(value="X --    Y --    Z --")

        root.title("MocapApi Joint Inspector")
        root.geometry("1050x600")
        root.minsize(760, 430)
        root.protocol("WM_DELETE_WINDOW", self.close)

        outer = ttk.Frame(root, padding=10)
        outer.pack(fill="both", expand=True)
        left = ttk.Frame(outer, width=220)
        left.pack(side="left", fill="y", padx=(0, 10))
        right = ttk.Frame(outer)
        right.pack(side="right", fill="both", expand=True)

        ttk.Label(left, text="수신 관절", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        ttk.Label(left, text="관절을 클릭하면 그래프가 바뀝니다.").pack(anchor="w", pady=(2, 8))
        list_frame = ttk.Frame(left)
        list_frame.pack(fill="both", expand=True)
        self.listbox = tk.Listbox(list_frame, width=27, exportselection=False)
        scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=scroll.set)
        self.listbox.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.listbox.bind("<<ListboxSelect>>", self._select_joint)

        top = ttk.Frame(right)
        top.pack(fill="x")
        self.title_label = ttk.Label(top, text="관절 데이터 대기 중", font=("Segoe UI", 16, "bold"))
        self.title_label.pack(side="left")
        ttk.Button(top, text="그래프 초기화", command=self.samples.clear).pack(side="right")
        ttk.Label(right, textvariable=self.value_text, font=("Consolas", 12)).pack(anchor="w", pady=(8, 5))

        self.canvas = tk.Canvas(right, bg="#111820", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        legend = ttk.Frame(right)
        legend.pack(fill="x", pady=(5, 0))
        for label, color in zip(LABELS, COLORS):
            tk.Label(legend, text=f"● {label}", fg=color, font=("Segoe UI", 10, "bold")).pack(side="left", padx=8)
        ttk.Label(legend, textvariable=self.status).pack(side="right")

        self._connect()
        root.after(10, self.update)

    def _connect(self):
        settings = mcp.MCPSettings()
        settings.set_udp(self.port)
        settings.set_bvh_data(mcp.MCPBvhData.Binary)
        settings.set_bvh_rotation(mcp.MCPBvhRotation.YXZ)
        self.app = mcp.MCPApplication()
        self.app.set_settings(settings)
        ok, message = self.app.open()
        if not ok:
            raise RuntimeError(f"UDP {self.port} open failed: {message}")
        self.status.set(f"UDP {self.port} 수신 대기")

    def _select_joint(self, _event=None):
        selection = self.listbox.curselection()
        if not selection:
            return
        name = self.listbox.get(selection[0])
        if name != self.selected.get():
            self.selected.set(name)
            self.title_label.configure(text=name)
            self.samples.clear()

    def _refresh_joint_list(self, names):
        old = self.selected.get()
        self.listbox.delete(0, "end")
        for name in names:
            self.listbox.insert("end", name)
        preferred = old if old in names else next((n for n in ("Spine3", "Spine2", "Spine1", "Spine") if n in names), names[0])
        index = names.index(preferred)
        self.listbox.selection_set(index)
        self.listbox.see(index)
        self.selected.set(preferred)
        self.title_label.configure(text=preferred)

    def update(self):
        try:
            updated = False
            for event in self.app.poll_next_event():
                if event.event_type == mcp.MCPEventType.AvatarUpdated:
                    self.avatar_handle = event.event_data.avatar_handle
                    updated = True
            if updated:
                avatar = mcp.MCPAvatar(self.avatar_handle)
                current = {joint.get_name(): joint for joint in avatar.get_joints()}
                if tuple(current) != tuple(self.joints):
                    self.joints = current
                    self._refresh_joint_list(sorted(current))
                name = self.selected.get()
                if name in current:
                    xyz = global_position(current[name])
                    now = time.monotonic()
                    self.samples.append((now, *xyz))
                    self.value_text.set(f"X {xyz[0]: .4f}    Y {xyz[1]: .4f}    Z {xyz[2]: .4f}")
                    self.frame_count += 1
                    if now - self.last_frame >= 1.0:
                        hz = self.frame_count / (now - self.last_frame)
                        self.status.set(f"UDP {self.port}  ● 수신 중  {hz:.1f} Hz  |  관절 {len(current)}개")
                        self.last_frame, self.frame_count = now, 0
            self.draw()
        except Exception as exc:
            self.status.set(f"오류: {exc}")
        finally:
            if self.app is not None:
                self.root.after(10, self.update)

    def draw(self):
        c = self.canvas
        c.delete("all")
        w, h = max(c.winfo_width(), 10), max(c.winfo_height(), 10)
        margin = 45
        c.create_rectangle(margin, 15, w - 15, h - 30, outline="#52606d")
        if len(self.samples) < 2:
            c.create_text(w / 2, h / 2, text="데이터 대기 중", fill="#c7d0d9", font=("Segoe UI", 14))
            return
        data = np.asarray(self.samples)
        cutoff = data[-1, 0] - self.seconds
        data = data[data[:, 0] >= cutoff]
        values = data[:, 1:]
        ymin, ymax = float(values.min()), float(values.max())
        pad = max((ymax - ymin) * 0.12, 1e-3)
        ymin, ymax = ymin - pad, ymax + pad
        for i in range(5):
            y = 15 + i * (h - 45) / 4
            value = ymax - i * (ymax - ymin) / 4
            c.create_line(margin, y, w - 15, y, fill="#263442")
            c.create_text(margin - 5, y, text=f"{value:.3f}", fill="#9aa8b5", anchor="e")
        t0, t1 = data[0, 0], max(data[-1, 0], data[0, 0] + 1e-6)
        for axis, color in enumerate(COLORS):
            points = []
            for row in data:
                x = margin + (row[0] - t0) / (t1 - t0) * (w - margin - 15)
                y = 15 + (ymax - row[axis + 1]) / (ymax - ymin) * (h - 45)
                points.extend((x, y))
            if len(points) >= 4:
                c.create_line(*points, fill=color, width=2, smooth=False)

    def close(self):
        app, self.app = self.app, None
        if app is not None:
            app.close()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="Live MocapApi joint XYZ GUI")
    parser.add_argument("--port", type=int, default=7002)
    parser.add_argument("--seconds", type=float, default=8.0, help="visible history")
    args = parser.parse_args()
    root = tk.Tk()
    MocapJointPlot(root, args.port, args.seconds)
    root.mainloop()


if __name__ == "__main__":
    main()
