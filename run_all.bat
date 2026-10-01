@echo off
REM Launches the three separate teleop processes, each in its own window,
REM all activated into the upperbody_teleop conda env:
REM   1. hand_process.py    -- Quest hand tracking -> DG5F finger retargeting + graph + recording
REM   2. mocap_process.py   -- Axis Studio/MocapApi -> arm/torso retargeting + graph + recording
REM   3. mujoco_physics_process.py -- receives both over UDP, physics-only simulation
REM
REM Edit the arguments below (Quest IP, mocap port, etc.) to match your setup.
REM Each window stays open after its process exits so errors are visible;
REM close the windows (or Ctrl+C inside each) to stop.
REM
REM --- scripts/hand_process.py (window 1) options ---
REM   --quest-host HOST / --quest-port PORT   Quest UDP source
REM   --send udp://HOST:PORT                  where to forward retargeted joints (-> process 3's --port)
REM   --hand-retargeter {anydex,dg5f}          which retargeter backend
REM   --anydex-config-left/--anydex-config-right PATH   AnyDex per-hand yaml config
REM   --record PATH                            record raw Quest hand landmarks to .jsonl
REM   --hz FLOAT                               poll/retarget rate (default 30)
REM
REM --- scripts/mocap_process.py (window 2) options ---
REM   --backend {mocapapi,replay}              live MocapApi UDP, or replay a recorded .jsonl
REM   --replay-file PATH / --no-loop           --backend replay: file to play (and whether to loop)
REM   --port INT                                MocapApi UDP port (live backend)
REM   --send udp://HOST:PORT                    -> process 3's --port
REM   --calibration-pose {attention,tpose,raise} pose used for calibrate()
REM   --torso {fixed,retarget}                  whether torso follows mocap or stays rigid
REM   --arm-solver {closed-form,mink}            IK backend for elbow+wrist
REM   --bvh-scale                                scale robot arm segments to the human's own proportions
REM   --record PATH                              record raw mocap landmarks to .jsonl
REM   --hz FLOAT                                 poll/retarget rate (default 30)
REM
REM --- scripts/mujoco_physics_process.py (window 3) options ---
REM   --port INT                                 UDP port both process 1 and 2 send to
REM   --hz FLOAT                                  physics step rate (default 60)
REM   --robot-only                                skip assets/scene/scene.xml (pedestal/table/cloth), robot only
REM   --diagnose-latency                          print network vs. actuator-tracking latency once a second
REM   --show-cameras                              open a window with cam_head/cam_left_wrist/cam_right_wrist
REM                                                (implies --capture-cameras)
REM   --capture-cameras                           render cameras headlessly (no window) for a future capture loop
REM   --camera-hz FLOAT                            camera render rate (default 3; renders on its own thread --
REM                                                ~7ms/frame on GPU, ~100-1000ms if it fell back to software GL)
REM   --gl-backend {auto,egl,glfw,osmesa}          camera renderer GL backend (auto: egl on Linux, glfw on Windows);
REM                                                startup log prints "camera GL renderer: ..." -- check it is your GPU
REM   --egl-device INT                             with egl: GPU index to render on
REM   --no-wrist-cameras                           with --show/--capture-cameras, render only cam_head
REM
REM press R inside the viewer window (process 3) to reset robot pose + cloth to initial state.

set CONDA_ROOT=C:\Users\admin\miniforge3
set ENV_NAME=upperbody_teleop
set PROJECT_ROOT=%~dp0

start "1: hand_process" cmd /k "call "%CONDA_ROOT%\Scripts\activate.bat" %ENV_NAME% && cd /d "%PROJECT_ROOT%" && python -m scripts.hand_process --quest-host 192.168.8.156 --quest-port 5005 --send udp://192.168.8.156:6100  --hand-retargeter anydex --anydex-config-left configs/anydex/dg5f_left_adaptive_quest3.yaml --anydex-config-right configs/anydex/dg5f_right_adaptive_quest3.yaml"

start "2: mocap_process" cmd /k "call "%CONDA_ROOT%\Scripts\activate.bat" %ENV_NAME% && cd /d "%PROJECT_ROOT%" && python -m scripts.mocap_process --port 7002 --send udp://192.168.8.156:6100"

@REM  start "3: mujoco_physics_process" cmd /k "call "%CONDA_ROOT%\Scripts\activate.bat" %ENV_NAME% && cd /d "%PROJECT_ROOT%" \ && python -m scripts.mujoco_physics_process --port 6100"
start "3: mujoco_physics_process" cmd /k "call "%CONDA_ROOT%\Scripts\activate.bat" %ENV_NAME% && cd /d "%PROJECT_ROOT%" \ && python -m scripts.mujoco_physics_process --port 6100 --capture-cameras --show-cameras --camera-hz 3 --diagnose-latency"
