"""Standalone SenseGlove-only monitor UI: confirms finger data is actually
arriving, with no MuJoCo/vedo/Perception Neuron dependency.

`scripts/vedo_local_preview.py` needs vedo (VTK/OpenGL) *and* a Perception
Neuron reader (`TeleopAggregator` won't produce a state until both PN and
SenseGlove have reported at least once), so it can't be used to check
SenseGlove alone while PN isn't wired up yet. This uses only stdlib
(`tkinter`), talks to `SenseGloveReaderBase` directly, and shows exactly
what matters for a hardware sanity check:

- per-hand connection status (LIVE / STALE / NO DATA), colored, with the
  age of the last sample -- this is the "is it actually receiving" signal,
  since `HandState` for a hand can arrive intermittently on flaky glove
  BLE links (see senseglove_bridge/README.md's connection-stability notes)
- 5 finger flexion bars (0=open .. 1=closed) per hand, live-updating

Usage:
    python scripts/senseglove_monitor.py --backend mock
    python scripts/senseglove_monitor.py --backend bridge   # real hardware, via senseglove_bridge.exe
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time
import tkinter as tk
from typing import Optional

_ROOT = pathlib.Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from devices.senseglove.sg_reader import (  # noqa: E402
    MockSenseGloveReader,
    SenseGloveJSONBridgeReader,
    SenseGloveReaderBase,
    SGCoreSenseGloveReader,
)
from devices.senseglove.sg_types import FINGER_NAMES, HandState  # noqa: E402

_LIVE_S = 0.3
_STALE_S = 1.5

_BAR_W = 220
_BAR_H = 22
_BG = "#1e1e1e"
_FG = "#e0e0e0"
_PANEL_BG = "#2a2a2a"
_TRACK_BG = "#3a3a3a"


def _status(age: Optional[float]) -> tuple[str, str]:
    """(text, color) for a hand's staleness."""
    if age is None:
        return "WAITING", "#888888"
    if age < _LIVE_S:
        return f"LIVE ({age * 1000:.0f} ms)", "#4caf50"
    if age < _STALE_S:
        return f"STALE ({age:.1f} s)", "#ff9800"
    return f"NO DATA ({age:.1f} s)", "#f44336"


def _bar_color(value: float) -> str:
    # green (open) -> amber -> red (closed), same hue ramp for both hands.
    r = int(80 + 150 * value)
    g = int(200 - 120 * value)
    return f"#{r:02x}{g:02x}60"


class HandPanel:
    def __init__(self, parent: tk.Widget, title: str):
        self.frame = tk.Frame(parent, bg=_PANEL_BG, padx=12, pady=10)
        tk.Label(
            self.frame, text=title, font=("Segoe UI", 14, "bold"), bg=_PANEL_BG, fg=_FG
        ).pack(anchor="w")
        self.status_label = tk.Label(
            self.frame, text="WAITING", font=("Segoe UI", 10), bg=_PANEL_BG, fg="#888888"
        )
        self.status_label.pack(anchor="w", pady=(0, 8))

        self.bars = {}
        self.value_labels = {}
        for name in FINGER_NAMES:
            row = tk.Frame(self.frame, bg=_PANEL_BG)
            row.pack(fill="x", pady=2)
            tk.Label(
                row, text=name.capitalize(), width=8, anchor="w",
                font=("Segoe UI", 10), bg=_PANEL_BG, fg=_FG,
            ).pack(side="left")
            canvas = tk.Canvas(row, width=_BAR_W, height=_BAR_H, bg=_TRACK_BG,
                                highlightthickness=0)
            canvas.pack(side="left", padx=6)
            bar = canvas.create_rectangle(0, 0, 0, _BAR_H, fill=_bar_color(0.0), width=0)
            self.bars[name] = (canvas, bar)
            value_label = tk.Label(row, text="0.00", width=5, font=("Segoe UI", 10),
                                    bg=_PANEL_BG, fg=_FG)
            value_label.pack(side="left")
            self.value_labels[name] = value_label

    def update(self, state: Optional[HandState], age: Optional[float]) -> None:
        text, color = _status(age)
        self.status_label.configure(text=text, fg=color)
        if state is None:
            return
        for i, name in enumerate(FINGER_NAMES):
            value = float(state.flexion[i])
            canvas, bar = self.bars[name]
            canvas.coords(bar, 0, 0, value * _BAR_W, _BAR_H)
            canvas.itemconfigure(bar, fill=_bar_color(value))
            self.value_labels[name].configure(text=f"{value:.2f}")


class MonitorApp:
    def __init__(self, reader: SenseGloveReaderBase, hz: float):
        self.reader = reader
        self.period_ms = int(1000.0 / hz)

        self.root = tk.Tk()
        self.root.title("SenseGlove Monitor")
        self.root.configure(bg=_BG)

        header = tk.Label(
            self.root, text="SenseGlove finger flexion (0 = open, 1 = closed)",
            font=("Segoe UI", 11), bg=_BG, fg=_FG, pady=8,
        )
        header.pack()

        body = tk.Frame(self.root, bg=_BG)
        body.pack(padx=12, pady=(0, 12))

        self.left_panel = HandPanel(body, "LEFT")
        self.left_panel.frame.pack(side="left", padx=(0, 10))
        self.right_panel = HandPanel(body, "RIGHT")
        self.right_panel.frame.pack(side="left")

        self._last_left: Optional[HandState] = None
        self._last_right: Optional[HandState] = None
        self._last_left_t: Optional[float] = None
        self._last_right_t: Optional[float] = None

    def _tick(self):
        left, right = self.reader.poll()
        now = time.time()
        if left is not None:
            self._last_left = left
            self._last_left_t = now
        if right is not None:
            self._last_right = right
            self._last_right_t = now

        left_age = (now - self._last_left_t) if self._last_left_t else None
        right_age = (now - self._last_right_t) if self._last_right_t else None
        self.left_panel.update(self._last_left, left_age)
        self.right_panel.update(self._last_right, right_age)

        self.root.after(self.period_ms, self._tick)

    def run(self):
        self.reader.connect()
        self.root.after(0, self._tick)
        self.root.mainloop()
        self.reader.close()


def main():
    parser = argparse.ArgumentParser(description="Standalone SenseGlove reception monitor (tkinter, no MuJoCo/vedo)")
    parser.add_argument("--backend", choices=["mock", "sgcore", "bridge"], default="mock")
    parser.add_argument("--bridge-host", default="127.0.0.1")
    parser.add_argument("--bridge-port", type=int, default=8850)
    parser.add_argument("--hz", type=float, default=30.0, help="UI refresh rate")
    args = parser.parse_args()

    if args.backend == "mock":
        reader: SenseGloveReaderBase = MockSenseGloveReader()
    elif args.backend == "sgcore":
        reader = SGCoreSenseGloveReader()
    else:
        reader = SenseGloveJSONBridgeReader(args.bridge_host, args.bridge_port)

    print(f"[senseglove_monitor] backend={args.backend} -- close the window to stop")
    app = MonitorApp(reader, args.hz)
    app.run()


if __name__ == "__main__":
    main()
