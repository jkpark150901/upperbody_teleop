"""Combined human teleop state: PN wrist/chest pose + SenseGlove hand state.

Mirrors the `HumanTeleopState` struct in plan section 6, minus the C++
syntax.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from devices.perception_neuron.pn_reader import PNReaderBase
from devices.senseglove.sg_reader import SenseGloveReaderBase
from devices.senseglove.sg_types import HandState
from teleop.geometry import Pose


@dataclass
class HumanTeleopState:
    timestamp: float
    chest: Pose
    left_wrist: Pose
    right_wrist: Pose
    left_hand: HandState
    right_hand: HandState

    # bookkeeping for latency / drop diagnostics (plan section 20, "Teleoperation latency")
    pn_age: float = 0.0
    sg_age: float = 0.0


class TeleopAggregator:
    """Polls PN + SenseGlove readers and merges them into HumanTeleopState.

    Uses a "hold last known sample" strategy: each stream is polled every
    tick; if a reader has nothing new, the previous sample for that stream
    is reused and its age is tracked so staleness is visible to the caller
    (and can be recorded — plan section 12/20).
    """

    def __init__(
        self,
        pn_reader: PNReaderBase,
        sg_reader: SenseGloveReaderBase,
        stale_warn_s: float = 0.25,
    ):
        self._pn = pn_reader
        self._sg = sg_reader
        self._stale_warn_s = stale_warn_s

        self._last_pn = None
        self._last_pn_t = None
        self._last_left_hand: Optional[HandState] = None
        self._last_right_hand: Optional[HandState] = None
        self._last_hand_t = None

    def connect(self) -> None:
        self._pn.connect()
        self._sg.connect()

    def close(self) -> None:
        self._pn.close()
        self._sg.close()

    def poll(self) -> Optional[HumanTeleopState]:
        now = time.time()

        pn_frame = self._pn.poll()
        if pn_frame is not None:
            self._last_pn = pn_frame
            self._last_pn_t = now

        left, right = self._sg.poll()
        if left is not None:
            self._last_left_hand = left
        if right is not None:
            self._last_right_hand = right
        if left is not None or right is not None:
            self._last_hand_t = now

        if self._last_pn is None or self._last_left_hand is None or self._last_right_hand is None:
            return None  # not enough data yet to form a full state

        pn_age = now - self._last_pn_t if self._last_pn_t else float("inf")
        sg_age = now - self._last_hand_t if self._last_hand_t else float("inf")

        if pn_age > self._stale_warn_s or sg_age > self._stale_warn_s:
            print(
                f"[teleop_aggregator] WARNING stale input: pn_age={pn_age:.3f}s "
                f"sg_age={sg_age:.3f}s"
            )

        return HumanTeleopState(
            timestamp=now,
            chest=self._last_pn.chest,
            left_wrist=self._last_pn.left_wrist,
            right_wrist=self._last_pn.right_wrist,
            left_hand=self._last_left_hand,
            right_hand=self._last_right_hand,
            pn_age=pn_age,
            sg_age=sg_age,
        )
