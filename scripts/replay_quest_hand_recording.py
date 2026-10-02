"""Replay a hand_process.py --record Quest/OpenXR hand skeleton JSONL.

This is for checking the *raw* Quest skeleton that arrived over UDP, before
AnyDex/analytic retargeting and before MuJoCo.  It shows the recorded
left/right 26-joint OpenXR skeletons and prints thumb-tip <-> index-tip
distance so pinch/contact can be checked against what the headset view shows.

Example:
    python -m scripts.replay_quest_hand_recording --file recordings/quest_hand_latest.jsonl
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Dict, Iterable, Optional

import numpy as np

_ROOT = pathlib.Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from devices.quest_hand.quest_hand_reader import (  # noqa: E402
    N_XR_JOINTS,
    XR_PALM,
    XR_WRIST,
    _CHAIN,
)
from scripts.skeleton_graph import SkeletonGraphWindow  # noqa: E402

_SIDES = ("left", "right")
_SIDE_OFFSET = {
    "left": np.array([0.0, 0.14, 0.0]),
    "right": np.array([0.0, -0.14, 0.0]),
}
_THUMB_TIP = 5
_INDEX_TIP = 10


def _graph_names_edges() -> tuple[list[str], list[tuple[str, str]]]:
    names: list[str] = []
    edges: list[tuple[str, str]] = []
    for side in _SIDES:
        side_names = [f"{side}_xr_{i}" for i in range(N_XR_JOINTS)]
        names.extend(side_names)
        edges.append((f"{side}_xr_{XR_WRIST}", f"{side}_xr_{XR_PALM}"))
        for chain in _CHAIN.values():
            edges.append((f"{side}_xr_{XR_WRIST}", f"{side}_xr_{chain[0]}"))
            edges.extend((f"{side}_xr_{a}", f"{side}_xr_{b}") for a, b in zip(chain, chain[1:]))
    return names, edges


def _load_frames(path: pathlib.Path) -> list[dict]:
    frames = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("type") == "frame":
                frames.append(record)
    if not frames:
        raise RuntimeError(f"{path}: no frame records found")
    return frames


def _positions_from_frame(frame: dict) -> tuple[Dict[str, np.ndarray], dict[str, Optional[float]]]:
    positions: Dict[str, np.ndarray] = {}
    pinch_dist: dict[str, Optional[float]] = {"left": None, "right": None}
    xr_by_side = frame.get("xr_joints", {})
    for side in _SIDES:
        raw = xr_by_side.get(side)
        if raw is None:
            continue
        xr = np.asarray(raw, dtype=float)
        if xr.shape[0] != N_XR_JOINTS or xr.shape[1] < 3:
            continue
        wrist = xr[XR_WRIST, :3]
        pts = xr[:, :3] - wrist + _SIDE_OFFSET[side]
        for i, p in enumerate(pts):
            positions[f"{side}_xr_{i}"] = p
        pinch_dist[side] = float(np.linalg.norm(xr[_THUMB_TIP, :3] - xr[_INDEX_TIP, :3]))
    return positions, pinch_dist


def _iter_frame_indices(num_frames: int, loop: bool) -> Iterable[int]:
    while True:
        for i in range(num_frames):
            yield i
        if not loop:
            return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=pathlib.Path, required=True,
                        help="hand_process.py --record JSONL file")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--labels", action="store_true",
                        help="show per-joint labels; off by default so fingertips remain visible")
    parser.add_argument("--print-hz", type=float, default=10.0,
                        help="max rate for console thumb/index distance logs")
    args = parser.parse_args()

    frames = _load_frames(args.file)
    names, edges = _graph_names_edges()
    graph = SkeletonGraphWindow(
        f"Quest hand recording: {args.file.name}",
        names,
        edges,
        show_labels=args.labels,
    )
    print(f"[replay_quest_hand_recording] loaded {len(frames)} frames from {args.file}")
    print("[replay_quest_hand_recording] distances are raw Quest thumb tip <-> index tip in metres")

    last_log = 0.0
    try:
        prev_t = float(frames[0].get("t", 0.0))
        for idx in _iter_frame_indices(len(frames), args.loop):
            frame = frames[idx]
            t = float(frame.get("t", prev_t))
            if idx > 0:
                dt = max(0.0, (t - prev_t) / max(args.speed, 1e-6))
                time.sleep(dt)
            prev_t = t

            positions, pinch_dist = _positions_from_frame(frame)
            graph.update(positions)
            now = time.monotonic()
            if now - last_log >= 1.0 / max(args.print_hz, 1e-6):
                last_log = now
                l = pinch_dist["left"]
                r = pinch_dist["right"]
                l_s = "NA" if l is None else f"{l * 1000.0:5.1f}mm"
                r_s = "NA" if r is None else f"{r * 1000.0:5.1f}mm"
                print(f"[quest-skel] frame={idx:05d} t={t:7.3f}s left={l_s} right={r_s}")
    except KeyboardInterrupt:
        pass
    finally:
        graph.close()


if __name__ == "__main__":
    main()
