@echo off
REM Local MuJoCo OK-sign test: no Quest, no mocap.
REM Window 1 runs physics/viewer. Window 2 sends a scripted OK-sign hand pose.

set CONDA_ROOT=C:\Users\admin\miniforge3
set ENV_NAME=upperbody_teleop
set PROJECT_ROOT=%~dp0

start "1: mujoco_physics_process" cmd /k "call "%CONDA_ROOT%\Scripts\activate.bat" %ENV_NAME% && cd /d "%PROJECT_ROOT%" && python -m scripts.mujoco_physics_process --port 6100 --diagnose-latency --sim-config configs/sim_params.yaml"

timeout /t 2 /nobreak >nul

start "2: ok_sign_sender" cmd /k "call "%CONDA_ROOT%\Scripts\activate.bat" %ENV_NAME% && cd /d "%PROJECT_ROOT%" && python -m scripts.ok_sign_sender --send udp://127.0.0.1:6100 --hand both --strength 1.0 --close-time 0.35 --hold-time 0.80 --open-time 0.35 --open-hold-time 0.50"
