"""Standalone Quest hand-tracking reception monitor -- no vedo/VTK, no URDF,
just "is anything arriving, from where, and is each hand actually tracked".
Same role as scripts/senseglove_monitor.py, extended with the connection-
level diagnostics a UDP sender needs that a HandState alone can't show.

A bare HandState (or None) can't tell two very different problems apart:
  - nothing is arriving at all (headset asleep, app not focused, wrong IP/
    port baked into the APK, different LAN, Windows Firewall) -- fix on the
    Quest/network side
  - packets ARE arriving but this hand reads inactive/untracked (out of
    camera view, poor lighting, hand tracking not permitted in the OS
    settings) -- fix on the Quest/user side, network is fine
This reads devices/quest_hand/quest_hand_reader.QuestHandUDPReader.stats
(ReceiveStats) to show both, plus which IP is actually sending (confirms
it's the headset you think it is) and a raw/malformed packet count (garbage
on the port, or a wire-format mismatch after an APK rebuild).

Usage:
    python scripts/quest_hand_monitor.py                    # binds 0.0.0.0:5005
    python scripts/quest_hand_monitor.py --port 5005
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

from devices.quest_hand.quest_hand_reader import QuestHandUDPReader  # noqa: E402
from devices.senseglove.sg_types import FINGER_NAMES, HandState  # noqa: E402

_LIVE_S = 0.3
_STALE_S = 1.5

_BAR_W = 220
_BAR_H = 22
_BG = "#1e1e1e"
_FG = "#e0e0e0"
_DIM_FG = "#909090"
_PANEL_BG = "#2a2a2a"
_TRACK_BG = "#3a3a3a"
_GREEN = "#4caf50"
_AMBER = "#ff9800"
_RED = "#f44336"
_GREY = "#888888"


def _hand_status(last_active: Optional[float], last_seen: Optional[float], now: float) -> tuple[str, str]:
    """(text, color) combining "is this hand tracked" (drives the flexion
    bars) with "is the headset sending this hand at all" -- the latter stays
    informative even while the former is stuck at NO DATA, which is exactly
    the case ("headset worn, nothing moves") this script exists to debug.
    """
    if last_active is not None and now - last_active < _LIVE_S:
        return f"LIVE ({(now - last_active) * 1000:.0f} ms)", _GREEN
    if last_seen is not None and now - last_seen < _LIVE_S:
        return f"SENDING, NOT TRACKED ({(now - last_seen) * 1000:.0f} ms)", _AMBER
    if last_active is not None and now - last_active < _STALE_S:
        return f"STALE ({now - last_active:.1f} s)", _AMBER
    if last_seen is not None and now - last_seen < _STALE_S:
        return f"STALE, NOT TRACKED ({now - last_seen:.1f} s)", _AMBER
    if last_seen is None:
        return "NO DATA", _RED
    return f"NO DATA ({now - last_seen:.1f} s)", _RED


def _bar_color(value: float) -> str:
    r = int(80 + 150 * value)
    g = int(200 - 120 * value)
    return f"#{r:02x}{g:02x}60"


class ConnectionPanel:
    """Transport-level view: is *anything* arriving on the socket, from
    where, and how much of it is garbage -- independent of either hand.
    """

    def __init__(self, parent: tk.Widget, listen_addr: str):
        self.frame = tk.Frame(parent, bg=_PANEL_BG, padx=12, pady=10)
        tk.Label(
            self.frame, text=f"Listening on {listen_addr}",
            font=("Segoe UI", 12, "bold"), bg=_PANEL_BG, fg=_FG,
        ).pack(anchor="w")
        self.rows: dict[str, tk.Label] = {}
        for key, label in (
            ("status", "Status"),
            ("sender", "Last sender"),
            ("rate", "Packet rate"),
            ("counts", "Total / malformed"),
        ):
            row = tk.Frame(self.frame, bg=_PANEL_BG)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=label, width=16, anchor="w", font=("Segoe UI", 10),
                      bg=_PANEL_BG, fg=_DIM_FG).pack(side="left")
            value = tk.Label(row, text="--", anchor="w", font=("Segoe UI", 10, "bold"),
                              bg=_PANEL_BG, fg=_FG)
            value.pack(side="left")
            self.rows[key] = value

        self._prev_total = 0
        self._prev_t = time.time()

    def update(self, stats, now: float) -> None:
        if stats.packets_total == 0:
            self.rows["status"].configure(text="NO PACKETS YET", fg=_RED)
        elif now - stats.last_packet_t < _LIVE_S:
            self.rows["status"].configure(text="RECEIVING", fg=_GREEN)
        elif now - stats.last_packet_t < _STALE_S:
            self.rows["status"].configure(text="STALE", fg=_AMBER)
        else:
            self.rows["status"].configure(
                text=f"SILENT ({now - stats.last_packet_t:.1f} s)", fg=_RED
            )

        sender = f"{stats.last_addr[0]}:{stats.last_addr[1]}" if stats.last_addr else "--"
        self.rows["sender"].configure(text=sender, fg=_FG)

        dt = now - self._prev_t
        rate = (stats.packets_total - self._prev_total) / dt if dt > 0 else 0.0
        self._prev_total, self._prev_t = stats.packets_total, now
        self.rows["rate"].configure(text=f"{rate:.0f} pkt/s", fg=_FG)

        malformed_fg = _RED if stats.packets_malformed else _DIM_FG
        self.rows["counts"].configure(
            text=f"{stats.packets_total} / {stats.packets_malformed}", fg=malformed_fg
        )


class HandPanel:
    def __init__(self, parent: tk.Widget, title: str):
        self.frame = tk.Frame(parent, bg=_PANEL_BG, padx=12, pady=10)
        tk.Label(
            self.frame, text=title, font=("Segoe UI", 14, "bold"), bg=_PANEL_BG, fg=_FG
        ).pack(anchor="w")
        self.status_label = tk.Label(
            self.frame, text="NO DATA", font=("Segoe UI", 10), bg=_PANEL_BG, fg=_GREY
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

    def update(self, state: Optional[HandState], last_active: Optional[float],
               last_seen: Optional[float], now: float) -> None:
        text, color = _hand_status(last_active, last_seen, now)
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
    def __init__(self, reader: QuestHandUDPReader, hz: float, listen_addr: str):
        self.reader = reader
        self.period_ms = int(1000.0 / hz)

        self.root = tk.Tk()
        self.root.title("Quest Hand Tracking Monitor")
        self.root.configure(bg=_BG)

        tk.Label(
            self.root, text="Quest 3 hand tracking reception (raw UDP -> flexion)",
            font=("Segoe UI", 11), bg=_BG, fg=_FG, pady=8,
        ).pack()

        self.conn_panel = ConnectionPanel(self.root, listen_addr)
        self.conn_panel.frame.pack(padx=12, pady=(0, 8), fill="x")

        body = tk.Frame(self.root, bg=_BG)
        body.pack(padx=12, pady=(0, 12))

        self.left_panel = HandPanel(body, "LEFT")
        self.left_panel.frame.pack(side="left", padx=(0, 10))
        self.right_panel = HandPanel(body, "RIGHT")
        self.right_panel.frame.pack(side="left")

        self._last_state: dict[str, Optional[HandState]] = {"left": None, "right": None}

    def _tick(self):
        left, right = self.reader.poll()
        if left is not None:
            self._last_state["left"] = left
        if right is not None:
            self._last_state["right"] = right

        now = time.time()
        stats = self.reader.stats
        self.conn_panel.update(stats, now)
        self.left_panel.update(
            self._last_state["left"], stats.last_active_t["left"], stats.last_seen_t["left"], now
        )
        self.right_panel.update(
            self._last_state["right"], stats.last_active_t["right"], stats.last_seen_t["right"], now
        )

        self.root.after(self.period_ms, self._tick)

    def run(self):
        self.reader.connect()
        self.root.after(0, self._tick)
        self.root.mainloop()
        self.reader.close()


def main():
    parser = argparse.ArgumentParser(
        description="Standalone Quest hand-tracking reception monitor (tkinter, no vedo)"
    )
    parser.add_argument(
        "--host", default="192.168.8.156",
        help="local address to bind -- the actual adapter's IP, not 0.0.0.0 "
             "(see devices/quest_hand/quest_hand_reader.py's QuestHandUDPReader docstring)",
    )
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--hz", type=float, default=20.0, help="UI refresh rate")
    args = parser.parse_args()

    reader = QuestHandUDPReader(args.host, args.port)
    print(f"[quest_hand_monitor] binding {args.host}:{args.port} -- close the window to stop")
    app = MonitorApp(reader, args.hz, f"{args.host}:{args.port}")
    app.run()


if __name__ == "__main__":
    main()
