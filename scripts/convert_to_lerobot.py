"""mujoco_teaching_data_v1 (scripts/mujoco_physics_process.py's
TeachingDataRecorder output: frames.jsonl + images/<camera>/*.png) -> a
LeRobotDataset v3.0-shaped directory (teaching_data_generation_rule.md
section 1): meta/info.json, meta/stats.json, meta/tasks.parquet,
meta/episodes/chunk-000/file-000.parquet, data/chunk-000/file-000.parquet.

This does NOT import lerobot -- the real LeRobotDataset.create()/
add_frame()/save_episode()/finalize() API (section 5) is the authoritative
writer; it also handles things this doesn't (multi-file chunking past the
~100MB/~200MB thresholds, stats computed incrementally, video encoding).
What this DOES do, built directly from the doc's own section 1 layout and
section 2 field table since lerobot itself isn't installed here: write
real parquet files with pyarrow (a small, non-ML dependency, unlike
lerobot/torch) so "does parquet actually get produced, with the right
columns/dtypes/episode boundaries" is genuinely checked, not just
asserted. videos/<cam>/chunk-000/file-000.mp4 is NOT produced (no video
encoder installed -- cv2/imageio/ffmpeg are all absent from this env);
images stay as source PNGs instead, called out explicitly below. Treat
this as a best-effort reproduction of the documented shape for offline
validation, not a byte-exact stand-in for the real writer -- re-run the
real LeRobotDataset once lerobot is actually installed before trusting
this for training.

Run standalone against one or more recorded episode directories:
    python -m scripts.convert_to_lerobot recordings/dummy_ep1 recordings/dummy_ep2 --out recordings/lerobot_ds
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Dict, List

_ROOT = pathlib.Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import io  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from PIL import Image  # noqa: E402

# Forbidden per teaching_data_generation_rule.md section 4 -- lerobot
# generates these itself; a frame dict that includes them would conflict
# with (or silently shadow) what LeRobotDataset.add_frame() computes.
_FORBIDDEN_KEYS = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
_REQUIRED_KEYS = {"observation.state", "action", "task"}


class EpisodeFormatError(ValueError):
    pass


def load_episode(episode_dir: pathlib.Path) -> tuple[dict, List[dict]]:
    """One TeachingDataRecorder episode -> (header dict, list of raw frame
    records from frames.jsonl, images NOT yet loaded)."""
    path = episode_dir / "frames.jsonl"
    if not path.exists():
        raise EpisodeFormatError(f"{path}: not found")
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise EpisodeFormatError(f"{path}: empty")
    header = json.loads(lines[0])
    if header.get("type") != "header" or header.get("format") != "mujoco_teaching_data_v1":
        raise EpisodeFormatError(f"{path}: unrecognized header {header!r}")
    frames = [json.loads(line) for line in lines[1:]]
    if not frames:
        raise EpisodeFormatError(f"{path}: header but no frames")
    return header, frames


def build_episode_frames(episode_dir: pathlib.Path, header: dict, raw_frames: List[dict]) -> List[Dict]:
    """raw frames.jsonl records -> the exact dict shape teaching_data_
    generation_rule.md section 4 specifies per add_frame() call, with
    images actually decoded (uint8 HxWx3, RGB -- PNG decodes to RGB
    already, see that section's "OpenCV read -> must swap order" caveat,
    which doesn't apply here since nothing touched cv2)."""
    n_joints = len(header["joint_names"])
    out = []
    for rec in raw_frames:
        state = np.asarray(rec["observation.state"], dtype=np.float32)
        action = np.asarray(rec["action"], dtype=np.float32)
        if state.shape != (n_joints,):
            raise EpisodeFormatError(f"observation.state shape {state.shape}, expected ({n_joints},)")
        if action.shape != (n_joints,):
            raise EpisodeFormatError(f"action shape {action.shape}, expected ({n_joints},)")
        frame: Dict = {
            "observation.state": state,
            "action": action,
            "task": header["task"],
        }
        for cam_name, rel_path in rec["images"].items():
            img_path = episode_dir / rel_path
            img = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
            frame[f"observation.images.{cam_name}"] = img
        out.append(frame)
    return out


def validate_dataset(episodes: List[List[Dict]], headers: List[dict]) -> List[str]:
    """Returns a list of problems found (empty = clean). Checks section
    3/4's rules that are actually checkable offline: joint order/count
    consistency across episodes, forbidden/required keys, dtypes, camera
    set consistency, resolution consistency, and the "episode too short"
    / "task string must match verbatim across an episode" rules."""
    problems: List[str] = []

    joint_names_sets = {tuple(h["joint_names"]) for h in headers}
    if len(joint_names_sets) > 1:
        problems.append(
            f"joint_names differ across episodes ({len(joint_names_sets)} distinct orderings) -- "
            "section 3.1 requires one fixed order for the whole dataset"
        )

    camera_sets = {tuple(sorted(h["camera_names"])) for h in headers}
    if len(camera_sets) > 1:
        problems.append(f"camera_names differ across episodes: {camera_sets}")

    for ep_idx, (frames, header) in enumerate(zip(episodes, headers)):
        if len(frames) == 0:
            problems.append(f"episode {ep_idx} ({header.get('task')!r}): 0 frames")
            continue
        tasks = {f["task"] for f in frames}
        if len(tasks) > 1:
            problems.append(f"episode {ep_idx}: task string varies within one episode: {tasks}")

        n_joints = len(header["joint_names"])
        img_shapes = {}
        for i, f in enumerate(frames):
            for key in _FORBIDDEN_KEYS:
                if key in f:
                    problems.append(f"episode {ep_idx} frame {i}: forbidden key {key!r} present")
            missing = _REQUIRED_KEYS - f.keys()
            if missing:
                problems.append(f"episode {ep_idx} frame {i}: missing required key(s) {missing}")
            if f["observation.state"].dtype != np.float32:
                problems.append(f"episode {ep_idx} frame {i}: observation.state dtype {f['observation.state'].dtype}, want float32")
            if f["action"].dtype != np.float32:
                problems.append(f"episode {ep_idx} frame {i}: action dtype {f['action'].dtype}, want float32")
            if f["observation.state"].shape != (n_joints,):
                problems.append(f"episode {ep_idx} frame {i}: observation.state has {f['observation.state'].shape[0]} values, header says {n_joints} joints")
            for key, val in f.items():
                if not key.startswith("observation.images."):
                    continue
                cam = key[len("observation.images."):]
                if val.dtype != np.uint8 or val.ndim != 3 or val.shape[2] != 3:
                    problems.append(f"episode {ep_idx} frame {i} {key}: shape/dtype {val.shape}/{val.dtype}, want (H,W,3) uint8")
                img_shapes.setdefault(cam, set()).add(val.shape)
        for cam, shapes in img_shapes.items():
            if len(shapes) > 1:
                problems.append(f"episode {ep_idx}: {cam} resolution varies within episode: {shapes}")

    all_tasks = {tuple(f["task"] for f in frames)[0] if frames else None for frames in episodes}
    return problems


def _vector_stats(arr: np.ndarray) -> dict:
    """arr: (N, D) -- per-dimension mean/std/min/max, plus a single shared count."""
    return {
        "mean": arr.mean(axis=0).tolist(), "std": arr.std(axis=0).tolist(),
        "min": arr.min(axis=0).tolist(), "max": arr.max(axis=0).tolist(),
        "count": [int(arr.shape[0])],
    }


def _scalar_stats(arr: np.ndarray) -> dict:
    """arr: (N,) -- lerobot's own convention is [value] even for scalar features."""
    return {
        "mean": [float(arr.mean())], "std": [float(arr.std())],
        "min": [float(arr.min())], "max": [float(arr.max())],
        "count": [int(arr.shape[0])],
    }


def _image_stats(images: List[np.ndarray]) -> dict:
    """images: list of (H,W,3) uint8 -- per-CHANNEL stats over all pixels of all
    images, normalized to 0..1 first (tbd.md section 5), shape (3,1,1) each."""
    stacked = np.stack(images).astype(np.float32) / 255.0  # (N,H,W,3)
    per_channel = stacked.transpose(3, 0, 1, 2).reshape(3, -1)  # (3, N*H*W)
    return {
        "mean": per_channel.mean(axis=1).reshape(3, 1, 1).tolist(),
        "std": per_channel.std(axis=1).reshape(3, 1, 1).tolist(),
        "min": per_channel.min(axis=1).reshape(3, 1, 1).tolist(),
        "max": per_channel.max(axis=1).reshape(3, 1, 1).tolist(),
        "count": [int(per_channel.shape[1])],
    }


def write_lerobot_dataset(
    out_dir: pathlib.Path, episodes: List[List[Dict]], headers: List[dict], fps: float,
) -> None:
    """Writes meta/info.json, meta/stats.json, meta/tasks.parquet,
    meta/episodes/chunk-000/file-000.parquet, data/chunk-000/file-000.parquet
    -- single chunk/file only (fine at dummy/small scale; the real writer
    splits into more files past its own size thresholds). Matches the
    field-by-field corrections in recordings/tbd.md (written against a
    real `LeRobotDataset(...)` load attempt + comparison to lerobot/
    svla_so101_pickplace), not just teaching_data_generation_rule.md's
    original (looser) description:
      - images are dtype="image" (PNG bytes inline in the data parquet,
        as a {bytes, path} struct -- HuggingFace datasets' own Image
        feature convention), not "video" -- still no actual video file,
        see module docstring; this is what tbd.md's #2/#1 asked for
        instead.
      - meta/info.json carries codebase_version/chunks_size/data_path/
        splits/etc. and full feature entries for timestamp/frame_index/
        episode_index/index/task_index (tbd.md #1).
      - meta/episodes/... carries tasks (list), file locations, dataset-
        wide row range, and per-episode stats/<feature>/{min,max,mean,
        std,count} (tbd.md #3).
      - meta/tasks.parquet has task as the DataFrame index, task_index as
        the only column (tbd.md #4), written via pandas specifically for
        this (pyarrow alone doesn't reproduce pandas' index-in-parquet
        metadata convention).
      - meta/stats.json adds count everywhere and covers every feature,
        including images (channel-wise, 0..1) and the auto-generated
        scalar columns (tbd.md #5).
    """
    meta_dir = out_dir / "meta"
    (meta_dir / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (out_dir / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    joint_names = headers[0]["joint_names"]
    n_state = len(joint_names)
    camera_names = headers[0]["camera_names"]

    all_tasks = []
    seen_tasks = {}
    for header in headers:
        task = header["task"]
        if task not in seen_tasks:
            seen_tasks[task] = len(all_tasks)
            all_tasks.append(task)

    cols = {
        "observation.state": [], "action": [], "timestamp": [],
        "frame_index": [], "episode_index": [], "index": [], "task_index": [],
    }
    for cam in camera_names:
        cols[f"observation.images.{cam}"] = []
    images_by_cam = {cam: [] for cam in camera_names}  # for stats.json, pixel values not re-decoded from parquet
    episode_rows = {
        "episode_index": [], "tasks": [], "length": [],
        "data/chunk_index": [], "data/file_index": [],
        "dataset_from_index": [], "dataset_to_index": [],
        "meta/episodes/chunk_index": [], "meta/episodes/file_index": [],
    }
    # Per-episode, per-feature stat columns get added to episode_rows below
    # once we know the feature set (observation.state/action + cameras).
    stat_fields = ("min", "max", "mean", "std", "count")
    for feat in ("observation.state", "action", *(f"observation.images.{c}" for c in camera_names)):
        for sf in stat_fields:
            episode_rows[f"stats/{feat}/{sf}"] = []

    global_index = 0
    img_resolution = None
    for ep_idx, (frames, header) in enumerate(zip(episodes, headers)):
        task_index = seen_tasks[header["task"]]
        ep_from = global_index
        ep_state, ep_action = [], []
        ep_images = {cam: [] for cam in camera_names}
        for frame_idx, frame in enumerate(frames):
            state = frame["observation.state"]
            action = frame["action"]
            cols["observation.state"].append(state.tolist())
            cols["action"].append(action.tolist())
            cols["timestamp"].append(frame_idx / fps)
            cols["frame_index"].append(frame_idx)
            cols["episode_index"].append(ep_idx)
            cols["index"].append(global_index)
            cols["task_index"].append(task_index)
            ep_state.append(state)
            ep_action.append(action)
            for cam in camera_names:
                img = frame[f"observation.images.{cam}"]
                if img_resolution is None:
                    img_resolution = img.shape
                buf = io.BytesIO()
                Image.fromarray(img).save(buf, format="PNG")
                cols[f"observation.images.{cam}"].append({"bytes": buf.getvalue(), "path": None})
                images_by_cam[cam].append(img)
                ep_images[cam].append(img)
            global_index += 1

        episode_rows["episode_index"].append(ep_idx)
        episode_rows["tasks"].append([header["task"]])
        episode_rows["length"].append(len(frames))
        episode_rows["data/chunk_index"].append(0)
        episode_rows["data/file_index"].append(0)
        episode_rows["dataset_from_index"].append(ep_from)
        episode_rows["dataset_to_index"].append(global_index)
        episode_rows["meta/episodes/chunk_index"].append(0)
        episode_rows["meta/episodes/file_index"].append(0)
        ep_state_stats = _vector_stats(np.array(ep_state, dtype=np.float32))
        ep_action_stats = _vector_stats(np.array(ep_action, dtype=np.float32))
        for sf in stat_fields:
            episode_rows[f"stats/observation.state/{sf}"].append(ep_state_stats[sf])
            episode_rows[f"stats/action/{sf}"].append(ep_action_stats[sf])
        for cam in camera_names:
            ep_img_stats = _image_stats(ep_images[cam])
            for sf in stat_fields:
                episode_rows[f"stats/observation.images.{cam}/{sf}"].append(ep_img_stats[sf])

    state_type = pa.list_(pa.float32(), n_state)
    image_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    data_fields = {
        "observation.state": pa.array(cols["observation.state"], type=state_type),
        "action": pa.array(cols["action"], type=state_type),
        "timestamp": pa.array(cols["timestamp"], type=pa.float32()),
        "frame_index": pa.array(cols["frame_index"], type=pa.int64()),
        "episode_index": pa.array(cols["episode_index"], type=pa.int64()),
        "index": pa.array(cols["index"], type=pa.int64()),
        "task_index": pa.array(cols["task_index"], type=pa.int64()),
    }
    for cam in camera_names:
        data_fields[f"observation.images.{cam}"] = pa.array(
            cols[f"observation.images.{cam}"], type=image_type
        )
    pq.write_table(pa.table(data_fields), out_dir / "data" / "chunk-000" / "file-000.parquet")

    pq.write_table(pa.Table.from_pandas(pd.DataFrame(episode_rows), preserve_index=False),
                    meta_dir / "episodes" / "chunk-000" / "file-000.parquet")

    pd.DataFrame(
        {"task_index": range(len(all_tasks))},
        index=pd.Index(all_tasks, name="task"),
    ).to_parquet(meta_dir / "tasks.parquet")

    state_arr = np.array(cols["observation.state"], dtype=np.float32)
    action_arr = np.array(cols["action"], dtype=np.float32)
    stats = {
        "observation.state": _vector_stats(state_arr),
        "action": _vector_stats(action_arr),
        "timestamp": _scalar_stats(np.array(cols["timestamp"], dtype=np.float32)),
        "frame_index": _scalar_stats(np.array(cols["frame_index"], dtype=np.float64)),
        "episode_index": _scalar_stats(np.array(cols["episode_index"], dtype=np.float64)),
        "index": _scalar_stats(np.array(cols["index"], dtype=np.float64)),
        "task_index": _scalar_stats(np.array(cols["task_index"], dtype=np.float64)),
    }
    for cam in camera_names:
        stats[f"observation.images.{cam}"] = _image_stats(images_by_cam[cam])
    (meta_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    h, w = img_resolution[0], img_resolution[1]
    info = {
        "codebase_version": "v3.0",
        "robot_type": "rby1_dg5f",
        "fps": int(fps),
        "total_episodes": len(episodes),
        "total_frames": global_index,
        "total_tasks": len(all_tasks),
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": None,
        "splits": {"train": f"0:{len(episodes)}"},
        "features": {
            "observation.state": {"dtype": "float32", "shape": [n_state], "names": joint_names},
            "action": {"dtype": "float32", "shape": [n_state], "names": joint_names},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
            **{
                f"observation.images.{cam}": {
                    "dtype": "image", "shape": [h, w, 3], "names": ["height", "width", "channels"],
                }
                for cam in camera_names
            },
        },
    }
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode_dirs", type=pathlib.Path, nargs="+")
    parser.add_argument("--out", type=pathlib.Path, default=None,
                         help="write a LeRobotDataset-shaped directory here (parquet via pyarrow, "
                         "no video -- see module docstring). Skipped if omitted (validate only).")
    parser.add_argument("--fps", type=float, default=30.0, help="--out: fps recorded into meta/info.json")
    args = parser.parse_args()

    episodes: List[List[Dict]] = []
    headers: List[dict] = []
    for ep_dir in args.episode_dirs:
        header, raw_frames = load_episode(ep_dir)
        frames = build_episode_frames(ep_dir, header, raw_frames)
        episodes.append(frames)
        headers.append(header)
        print(f"[convert_to_lerobot] {ep_dir}: {len(frames)} frames, task={header['task']!r}, "
              f"cameras={header['camera_names']}, joints={len(header['joint_names'])}")

    problems = validate_dataset(episodes, headers)
    total_frames = sum(len(e) for e in episodes)
    print(f"\n[convert_to_lerobot] {len(episodes)} episode(s), {total_frames} frame(s) total")
    if problems:
        print(f"[convert_to_lerobot] FAILED -- {len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print("[convert_to_lerobot] OK -- frame dicts match teaching_data_generation_rule.md's schema.")

    if args.out is not None:
        write_lerobot_dataset(args.out, episodes, headers, args.fps)
        print(f"[convert_to_lerobot] wrote parquet dataset -> {args.out} "
              "(no videos/ -- no video encoder installed, see module docstring)")
    else:
        print("[convert_to_lerobot] --out not given, skipped writing parquet (validation only).")


if __name__ == "__main__":
    main()
