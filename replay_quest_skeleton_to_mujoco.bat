@echo off
REM Replay a recorded Quest/OpenXR hand skeleton through the configured hand
REM retargeter and view the resulting DG5F fingers in MuJoCo.

set CONDA_ROOT=C:\Users\admin\miniforge3
set ENV_NAME=upperbody_teleop
set PROJECT_ROOT=%~dp0
set RECORD_FILE=recordings\quest_hand_latest.jsonl
set PORT=6100

call "%CONDA_ROOT%\Scripts\activate.bat" %ENV_NAME%
cd /d "%PROJECT_ROOT%"

start "1: mujoco_physics_process quest replay" cmd /k "call "%CONDA_ROOT%\Scripts\activate.bat" %ENV_NAME% && cd /d "%PROJECT_ROOT%" && python -m scripts.mujoco_physics_process --port %PORT% --diagnose-latency --sim-config configs/sim_params.yaml"
timeout /t 2 /nobreak >nul
start "2: quest_skeleton_replay_sender" cmd /k "call "%CONDA_ROOT%\Scripts\activate.bat" %ENV_NAME% && cd /d "%PROJECT_ROOT%" && python -m scripts.replay_quest_hand_to_mujoco --file "%RECORD_FILE%" --send udp://127.0.0.1:%PORT% --sim-config configs/sim_params.yaml --source retargeted --loop"
