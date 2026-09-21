#!/usr/bin/env bash
# Opens RB-Y1 (rby1ub) + DG5F hands in an interactive MuJoCo viewer.
# Rebuild with build_dg5f_rby1.py if rby1ub_src or the DG5F URDFs change.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="/home/spatial/miniforge3/envs/humanoid_vla/bin/python"

export DISPLAY="${DISPLAY:-:1}"
export XAUTHORITY="${XAUTHORITY:-/run/user/1000/gdm/Xauthority}"
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export __NV_PRIME_RENDER_OFFLOAD=1

cd "$SCRIPT_DIR"
exec "$PY" run_dg5f_viewer.py
