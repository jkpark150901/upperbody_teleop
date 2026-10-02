#!/usr/bin/env bash
# Linux port of run_all.bat -- launches the three teleop processes (see
# that file's header comment for what each one does and the full option
# list) into the upperbody_teleop conda env. No window titles to kill by
# on Linux, so PIDs go to .run_all.pids for stop_all.sh instead.
#
# Edit the arguments below (Quest IP, mocap port, etc.) to match your
# setup. Logs go to logs/<name>.log; `tail -f logs/*.log` to watch them.
#
# --- scripts/hand_process.py options ---
#   --quest-host HOST / --quest-port PORT   Quest UDP source
#   --send udp://HOST:PORT                  where to forward retargeted joints (-> process 3's --port)
#   --hand-retargeter {anydex,dg5f}          which retargeter backend
#   --anydex-config-left/--anydex-config-right PATH   AnyDex per-hand yaml config
#   --record PATH                            record raw Quest hand landmarks to .jsonl
#   --hz FLOAT                               poll/retarget rate (default 30)
#   --arm-ik {none,quest}                    quest: also solve torso-fixed wrist-pos+orientation IK per
#                                             arm from the Quest wrist pose, sent alongside fingers --
#                                             a VR-only alternative to window 2's MocapApi arms. Type
#                                             'c' + Enter in this window to (re)calibrate (arms forward
#                                             90deg, backs of hands up, hold still). Don't run window 2
#                                             at the same time in this mode.
#   --calibration-pose {attention,tpose,raise}  --arm-ik quest: RBY1UpperBodyRetargeter calibration_pose
#   --wrist-scale FLOAT                      --arm-ik quest: Quest-wrist-delta -> robot-delta scale
#   --quest-max-reach FLOAT                  --arm-ik quest: max shoulder-to-wrist reach in metres
#
# --- scripts/mocap_process.py options ---
#   --backend {mocapapi,replay}              live MocapApi UDP, or replay a recorded .jsonl
#   --replay-file PATH / --no-loop           --backend replay: file to play (and whether to loop)
#   --port INT                                MocapApi UDP port (live backend)
#   --send udp://HOST:PORT                    -> process 3's --port
#   --calibration-pose {attention,tpose,raise} pose used for calibrate()
#   --torso {fixed,retarget}                  whether torso follows mocap or stays rigid
#   --arm-solver {closed-form,mink}            IK backend for elbow+wrist
#   --bvh-scale                                scale robot arm segments to the human's own proportions
#   --record PATH                              record raw mocap landmarks to .jsonl
#   --hz FLOAT                                 poll/retarget rate (default 30)
#
# --- scripts/mujoco_physics_process.py options ---
#   --port INT                                 UDP port both process 1 and 2 send to
#   --hz FLOAT                                  physics step rate (default 60)
#   --robot-only                                skip assets/scene/scene.xml (pedestal/table/cloth), robot only
#   --diagnose-latency                          print network vs. actuator-tracking latency once a second
#   --show-cameras                              open a window with cam_head/cam_left_wrist/cam_right_wrist
#                                                (implies --capture-cameras)
#   --capture-cameras                           render cameras headlessly (no window) for a future capture loop
#   --camera-hz FLOAT                            camera render rate (default: configs/sim_params.yaml's
#                                                camera.hz; renders on its own thread when the GL backend
#                                                allows it -- ~7ms/frame on GPU, ~100-1000ms if it fell
#                                                back to software GL)
#   --sim-config PATH                            actuator gains / cloth physics / camera resolution+rate,
#                                                read fresh every run (default: configs/sim_params.yaml --
#                                                edit that file and re-run to test new values)
#   --gl-backend {auto,egl,glfw,osmesa}          camera renderer GL backend (auto: egl on Linux, glfw on Windows);
#                                                startup log prints "camera GL renderer: ..." -- check it is your GPU
#   --egl-device INT                             with egl: GPU index to render on
#   --no-wrist-cameras                           with --show/--capture-cameras, render only cam_head
#   --record-dataset DIR --task "description"    recording SESSION base dir. Press S in the viewer
#                                                window to toggle start/stop of a new episode
#                                                (auto-numbered DIR/ep_000, ep_001, ...) -- typing
#                                                'record'/'r' or 'stop'/'s' into this console also
#                                                works, for when the viewer isn't focused. Each stop
#                                                also (re)converts every episode recorded so far into
#                                                DIR/lerobot_ds.
#   --replay-dataset PATH --replay-speed FLOAT --loop
#                                                play one recorded episode (e.g. DIR/ep_000) back
#                                                instead of listening on --port (use replay_dataset.sh
#                                                for this, not this script).
#
# press R inside the viewer window (process 3) to reset robot pose + cloth to initial state.
set -euo pipefail

CONDA_ROOT="${CONDA_ROOT:-$HOME/miniforge3}"
ENV_NAME=upperbody_teleop
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

: > .run_all.pids

nohup python -m scripts.hand_process \
  --quest-host 192.168.8.156 --quest-port 5005 --send udp://192.168.8.156:6100 \
  --hand-retargeter anydex \
  --anydex-config-left configs/anydex/dg5f_left_adaptive_quest3.yaml \
  --anydex-config-right configs/anydex/dg5f_right_adaptive_quest3.yaml \
  > logs/hand_process.log 2>&1 &
echo $! >> .run_all.pids

nohup python -m scripts.mocap_process --port 7002 --send udp://192.168.8.156:6100 \
  > logs/mocap_process.log 2>&1 &
echo $! >> .run_all.pids

nohup python -m scripts.mujoco_physics_process --port 6100 \
  --capture-cameras --show-cameras --camera-hz 10 --diagnose-latency \
  > logs/mujoco_physics_process.log 2>&1 &
echo $! >> .run_all.pids

echo "started: $(cat .run_all.pids | tr '\n' ' ')"
echo "logs: logs/hand_process.log logs/mocap_process.log logs/mujoco_physics_process.log"
echo "stop with: ./stop_all.sh"
