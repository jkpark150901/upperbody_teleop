#!/usr/bin/env bash
# Linux port of replay_dataset.bat -- plays back a record_dataset.sh
# episode: drives qpos straight from frames.jsonl (no physics/actuators)
# and shows its saved camera images next to the live viewer. See
# scripts/mujoco_physics_process.py's --replay-dataset help.
#
# Usage: ./replay_dataset.sh [dataset_dir] [replay_speed]
set -euo pipefail

CONDA_ROOT="${CONDA_ROOT:-$HOME/miniforge3}"
ENV_NAME=upperbody_teleop
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

DATASET_DIR="${1:-recordings/ep001}"
REPLAY_SPEED="${2:-1.0}"

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

python -m scripts.mujoco_physics_process \
  --replay-dataset "$DATASET_DIR" --replay-speed "$REPLAY_SPEED" --loop
