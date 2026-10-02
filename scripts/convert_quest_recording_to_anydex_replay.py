"""Convert hand_process.py Quest/OpenXR JSONL to AnyDex MediaPipe replay PKL.

AnyDex's example/input/mediapipe_replay.py expects:
    [{"t": float, "left_fingers": (21,3), "right_fingers": (21,3)}, ...]

This script converts our recorded OpenXR 26-joint arrays into the 21-keypoint
order used by AnyDex, so the same captured Quest motion can be tested against
AnyDex's built-in hands such as Inspire.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import pickle
import sys

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from robot_hand.anydex_retarget import openxr_to_mediapipe21  # noqa: E402

_SIDES = ("left", "right")


def _load_jsonl(path: pathlib.Path) -> list[dict]:
    frames: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("type") == "frame":
                frames.append(record)
    if not frames:
        raise RuntimeError(f"{path}: no frame records found")
    return frames


def _convert_side(raw) -> np.ndarray:
    if raw is None:
        return np.zeros((21, 3), dtype=np.float32)
    xr = np.asarray(raw, dtype=np.float32)
    if xr.shape[0] != 26 or xr.shape[1] < 3 or not np.isfinite(xr[:, :3]).all():
        return np.zeros((21, 3), dtype=np.float32)
    return openxr_to_mediapipe21(xr).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=pathlib.Path, default=pathlib.Path("recordings/quest_hand_latest.jsonl"))
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("third_party/AnyDexRetarget/example/data/quest_hand_latest_anydex.pkl"),
    )
    parser.add_argument("--start-time", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=None)
    args = parser.parse_args()

    frames_in = _load_jsonl(args.input)
    start_time = max(0.0, float(args.start_time))
    end_time = None if args.duration is None else start_time + max(0.0, float(args.duration))

    selected = [
        frame for frame in frames_in
        if float(frame.get("t", 0.0)) >= start_time
        and (end_time is None or float(frame.get("t", 0.0)) <= end_time)
    ]
    if not selected:
        raise RuntimeError(f"{args.input}: no frames in selected time window")

    t0 = float(selected[0].get("t", 0.0))
    frames_out: list[dict] = []
    counts = {"left": 0, "right": 0}
    for frame in selected:
        xr_by_side = frame.get("xr_joints", {})
        out = {"t": float(frame.get("t", 0.0)) - t0}
        for side in _SIDES:
            raw = xr_by_side.get(side)
            mp = _convert_side(raw)
            out[f"{side}_fingers"] = mp
            if raw is not None and np.any(mp):
                counts[side] += 1
        frames_out.append(out)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as f:
        pickle.dump(frames_out, f)

    print(f"[convert_quest_recording_to_anydex_replay] input={args.input}")
    print(f"[convert_quest_recording_to_anydex_replay] output={args.output}")
    print(
        f"[convert_quest_recording_to_anydex_replay] frames={len(frames_out)} "
        f"duration={frames_out[-1]['t']:.3f}s left_tracked={counts['left']} "
        f"right_tracked={counts['right']}"
    )


if __name__ == "__main__":
    main()
