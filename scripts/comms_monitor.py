"""Combined reception monitor: Quest hand-tracking UDP + MocapApi (Axis
Studio) body/finger data, in one tkinter window -- so "Quest hand data
isn't arriving" can be told apart from "mocap body isn't arriving" (or both,
or neither) at a glance, instead of guessing from the MuJoCo/vedo viewer
sitting frozen with no idea which upstream source is actually the problem.

Reuses quest_hand_monitor.py's ConnectionPanel/HandPanel as-is for the Quest
side; the MocapApi side gets its own MocapPanel (connection status: is
poll() actually returning frames) plus the same HandPanel widget fed from
MocapLandmarkReader.last_hand (the BVH finger-flexion path added for
--hand-backend none, see scripts/upper_body_retarget_vedo.py's
_bvh_hand_state) -- so this window also confirms that data, not just the
Quest wire.

Usage:
    python -m scripts.comms_monitor
    python -m scripts.comms_monitor --mocap-port 7002 --quest-host 192.168.8.156 --quest-port 5005
    python -m scripts.comms_monitor --no-quest   # mocap only
    python -m scripts.comms_monitor --no-mocap   # quest only
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
from scripts.quest_hand_monitor import (  # noqa: E402
    _AMBER, _BG, _DIM_FG, _FG, _GREEN, _LIVE_S, _PANEL_BG, _RED, _STALE_S,
    ConnectionPanel, HandPanel,
)

_MOCAP_STATUS_ROW_HEIGHT = 22


class MocapPanel:
    """Connection-level view for MocapApi: is poll() actually returning new
    frames, and (if not) why -- an import/open() failure reads very
    differently (ERROR, with the exception text) from "connected but the
    stream just went quiet" (SILENT)."""

    def __init__(self, parent: tk.Widget, port: int):
        self.frame = tk.Frame(parent, bg=_PANEL_BG, padx=12, pady=10)
        tk.Label(
            self.frame, text=f"MocapApi UDP {port}",
            font=("Segoe UI", 12, "bold"), bg=_PANEL_BG, fg=_FG,
        ).pack(anchor="w")
        self.status_label = tk.Label(
            self.frame, text="CONNECTING...", font=("Segoe UI", 10, "bold"),
            bg=_PANEL_BG, fg=_DIM_FG, wraplength=380, justify="left",
        )
        self.status_label.pack(anchor="w", pady=(2, 0))

    def update(self, last_frame_t: Optional[float], error: Optional[str], now: float) -> None:
        if error is not None:
            self.status_label.configure(text=f"ERROR: {error}", fg=_RED)
        elif last_frame_t is None:
            self.status_label.configure(text="NO FRAMES YET", fg=_RED)
        elif now - last_frame_t < _LIVE_S:
            self.status_label.configure(text="RECEIVING", fg=_GREEN)
        elif now - last_frame_t < _STALE_S:
            self.status_label.configure(text=f"STALE ({now - last_frame_t:.1f} s)", fg=_AMBER)
        else:
            self.status_label.configure(text=f"SILENT ({now - last_frame_t:.1f} s)", fg=_RED)


class MocapSource:
    """Owns the MocapLandmarkReader so import/open() failures (no MocapApi
    SDK, wrong port, Axis Studio not streaming) surface as a status string
    instead of crashing this whole monitor window."""

    def __init__(self, port: int):
        self.port = port
        self.reader = None
        self.error: Optional[str] = None
        self.last_frame_t: Optional[float] = None

    def connect(self) -> None:
        try:
            from scripts.upper_body_retarget_vedo import MocapLandmarkReader
            self.reader = MocapLandmarkReader(self.port)
            self.reader.connect()
        except Exception as e:  # noqa: BLE001 -- surfaced in the UI, not swallowed
            self.error = str(e)

    def poll(self) -> None:
        if self.reader is None:
            return
        try:
            frame = self.reader.poll()
        except Exception as e:  # noqa: BLE001
            self.error = str(e)
            return
        if frame is not None:
            self.last_frame_t = time.time()

    @property
    def last_hand(self) -> dict:
        if self.reader is None:
            return {"left": None, "right": None}
        return self.reader.last_hand


class MonitorApp:
    def __init__(
        self,
        quest_reader: Optional[QuestHandUDPReader],
        mocap: Optional[MocapSource],
        hz: float,
        quest_listen_addr: str,
    ):
        self.quest_reader = quest_reader
        self.mocap = mocap
        self.period_ms = int(1000.0 / hz)

        self.root = tk.Tk()
        self.root.title("Comms Monitor (Quest hand + MocapApi)")
        self.root.configure(bg=_BG)

        if quest_reader is not None:
            tk.Label(
                self.root, text="Quest 3 hand tracking reception",
                font=("Segoe UI", 11, "bold"), bg=_BG, fg=_FG, pady=(8, 2),
            ).pack()
            self.quest_conn_panel = ConnectionPanel(self.root, quest_listen_addr)
            self.quest_conn_panel.frame.pack(padx=12, pady=(0, 4), fill="x")
            quest_body = tk.Frame(self.root, bg=_BG)
            quest_body.pack(padx=12, pady=(0, 10))
            self.quest_left_panel = HandPanel(quest_body, "LEFT (Quest)")
            self.quest_left_panel.frame.pack(side="left", padx=(0, 10))
            self.quest_right_panel = HandPanel(quest_body, "RIGHT (Quest)")
            self.quest_right_panel.frame.pack(side="left")
            self._quest_last_state: dict = {"left": None, "right": None}

        if mocap is not None:
            tk.Label(
                self.root, text="MocapApi (Axis Studio) reception",
                font=("Segoe UI", 11, "bold"), bg=_BG, fg=_FG, pady=(4, 2),
            ).pack()
            self.mocap_panel = MocapPanel(self.root, mocap.port)
            self.mocap_panel.frame.pack(padx=12, pady=(0, 4), fill="x")
            mocap_body = tk.Frame(self.root, bg=_BG)
            mocap_body.pack(padx=12, pady=(0, 12))
            self.mocap_left_panel = HandPanel(mocap_body, "LEFT (BVH fingers)")
            self.mocap_left_panel.frame.pack(side="left", padx=(0, 10))
            self.mocap_right_panel = HandPanel(mocap_body, "RIGHT (BVH fingers)")
            self.mocap_right_panel.frame.pack(side="left")

    def _tick(self):
        now = time.time()

        if self.quest_reader is not None:
            left, right = self.quest_reader.poll()
            if left is not None:
                self._quest_last_state["left"] = left
            if right is not None:
                self._quest_last_state["right"] = right
            stats = self.quest_reader.stats
            self.quest_conn_panel.update(stats, now)
            self.quest_left_panel.update(
                self._quest_last_state["left"], stats.last_active_t["left"], stats.last_seen_t["left"], now
            )
            self.quest_right_panel.update(
                self._quest_last_state["right"], stats.last_active_t["right"], stats.last_seen_t["right"], now
            )

        if self.mocap is not None:
            self.mocap.poll()
            self.mocap_panel.update(self.mocap.last_frame_t, self.mocap.error, now)
            hand = self.mocap.last_hand
            # No separate connection-vs-tracked distinction on this side
            # (MocapApi either gives you the frame or it doesn't) -- reuse
            # HandPanel with last_frame_t standing in for both timestamps.
            self.mocap_left_panel.update(hand["left"], self.mocap.last_frame_t, self.mocap.last_frame_t, now)
            self.mocap_right_panel.update(hand["right"], self.mocap.last_frame_t, self.mocap.last_frame_t, now)

        self.root.after(self.period_ms, self._tick)

    def run(self):
        if self.quest_reader is not None:
            self.quest_reader.connect()
        if self.mocap is not None:
            self.mocap.connect()
        self.root.after(0, self._tick)
        self.root.mainloop()
        if self.quest_reader is not None:
            self.quest_reader.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quest-host", default="192.168.8.156")
    parser.add_argument("--quest-port", type=int, default=5005)
    parser.add_argument("--mocap-port", type=int, default=7002)
    parser.add_argument("--hz", type=float, default=20.0, help="UI refresh rate")
    parser.add_argument("--no-quest", action="store_true", help="mocap only")
    parser.add_argument("--no-mocap", action="store_true", help="quest only")
    args = parser.parse_args()
    if args.no_quest and args.no_mocap:
        parser.error("can't pass both --no-quest and --no-mocap -- nothing left to monitor")

    quest_reader = None if args.no_quest else QuestHandUDPReader(args.quest_host, args.quest_port)
    mocap = None if args.no_mocap else MocapSource(args.mocap_port)

    print(
        "[comms_monitor] "
        + (f"quest={args.quest_host}:{args.quest_port} " if quest_reader else "")
        + (f"mocap_port={args.mocap_port}" if mocap else "")
        + " -- close the window to stop"
    )
    app = MonitorApp(quest_reader, mocap, args.hz, f"{args.quest_host}:{args.quest_port}")
    app.run()


if __name__ == "__main__":
    main()
