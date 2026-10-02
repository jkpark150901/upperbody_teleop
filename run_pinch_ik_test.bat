@echo off
REM Local MuJoCo thumb+index IK pinch test: no Quest, no mocap, no AnyDex.

set CONDA_ROOT=C:\Users\admin\miniforge3
set ENV_NAME=upperbody_teleop
set PROJECT_ROOT=%~dp0
set PORT=6100

start "1: mujoco_physics_process pinch ik" cmd /k "call "%CONDA_ROOT%\Scripts\activate.bat" %ENV_NAME% && cd /d "%PROJECT_ROOT%" && python -m scripts.mujoco_physics_process --port %PORT% --diagnose-latency --sim-config configs/sim_params.yaml"

powershell -NoProfile -Command "Start-Sleep -Seconds 2"

start "2: pinch_ik_sender" cmd /k "call "%CONDA_ROOT%\Scripts\activate.bat" %ENV_NAME% && cd /d "%PROJECT_ROOT%" && python -m scripts.pinch_motion_sender --send udp://127.0.0.1:%PORT% --hand both --strength 1.0 --static"
