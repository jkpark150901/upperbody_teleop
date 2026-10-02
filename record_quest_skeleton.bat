@echo off
REM Record raw Quest/OpenXR hand skeletons from hand_process.py.
REM Output: recordings\quest_hand_latest.jsonl

set CONDA_ROOT=C:\Users\admin\miniforge3
set ENV_NAME=upperbody_teleop
set PROJECT_ROOT=%~dp0
set RECORD_FILE=recordings\quest_hand_latest.jsonl

if not exist "%PROJECT_ROOT%recordings" mkdir "%PROJECT_ROOT%recordings"

echo Recording Quest hand skeletons to %RECORD_FILE%
echo Close this window or press Ctrl+C to stop recording.

call "%CONDA_ROOT%\Scripts\activate.bat" %ENV_NAME%
cd /d "%PROJECT_ROOT%"
python -m scripts.hand_process --quest-host 192.168.8.156 --quest-port 5005 --send udp://127.0.0.1:6100 --sim-config configs/sim_params.yaml --record "%RECORD_FILE%" --no-pinch-preset-grip
