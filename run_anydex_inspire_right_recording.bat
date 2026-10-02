@echo off
REM Test AnyDex Inspire Hand with the current Quest recording, right hand.

set CONDA_ROOT=C:\Users\admin\miniforge3
set ENV_NAME=upperbody_teleop
set PROJECT_ROOT=%~dp0
set RECORD_PKL=data\quest_hand_latest_first10_anydex.pkl

call "%CONDA_ROOT%\Scripts\activate.bat" %ENV_NAME%
cd /d "%PROJECT_ROOT%"
python -m scripts.run_anydex_hand_recording --config config/adaptive/quest3/quest3_inspire_hand.yaml --hand right --play %RECORD_PKL% --speed 1.0 --loop
