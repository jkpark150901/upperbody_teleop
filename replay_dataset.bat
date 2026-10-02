@echo off
REM Plays back a record_dataset.bat episode: drives qpos straight from
REM frames.jsonl (no physics/actuators) and shows its saved camera images
REM next to the live viewer. See scripts/mujoco_physics_process.py's
REM --replay-dataset help.
REM
REM Edit DATASET_DIR below, or run `replay_dataset.bat <dir> [--loop]`.

set CONDA_ROOT=C:\Users\admin\miniforge3
set ENV_NAME=upperbody_teleop
set PROJECT_ROOT=%~dp0

set DATASET_DIR=recordings\ep001
set REPLAY_SPEED=1.0
if not "%~1"=="" set DATASET_DIR=%~1

call "%CONDA_ROOT%\Scripts\activate.bat" %ENV_NAME%
cd /d "%PROJECT_ROOT%"
python -m scripts.mujoco_physics_process --replay-dataset "%DATASET_DIR%" --replay-speed %REPLAY_SPEED% --loop
