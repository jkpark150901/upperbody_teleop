#!/usr/bin/env bash
# Linux port of record_dataset.bat -- records one teaching-data episode
# with scripts/mujoco_physics_process.py (frames.jsonl + images/<camera>/
# *.png, synced to the camera's own render rate -- see
# teaching_data_generation_rule.md). Drive the robot via run_all.sh's
# process 1/2 as usual while this is running; Ctrl+C (or close the viewer
# window) to stop and finalize the recording.
#
# Usage: ./record_dataset.sh [dataset_dir] ["task description"]
set -euo pipefail

CONDA_ROOT="${CONDA_ROOT:-$HOME/miniforge3}"
ENV_NAME=upperbody_teleop
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

DATASET_DIR="${1:-recordings/ep001}"
TASK="${2:-pick up the cloth}"

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

python -m scripts.mujoco_physics_process --port 6100 \
  --record-dataset "$DATASET_DIR" --task "$TASK"
