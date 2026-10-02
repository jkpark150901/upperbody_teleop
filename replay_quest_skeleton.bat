@echo off
REM Replay raw Quest/OpenXR hand skeletons recorded by record_quest_skeleton.bat.

set CONDA_ROOT=C:\Users\admin\miniforge3
set ENV_NAME=upperbody_teleop
set PROJECT_ROOT=%~dp0
set RECORD_FILE=recordings\quest_hand_latest.jsonl

call "%CONDA_ROOT%\Scripts\activate.bat" %ENV_NAME%
cd /d "%PROJECT_ROOT%"
python -m scripts.replay_quest_hand_recording --file "%RECORD_FILE%" --loop
