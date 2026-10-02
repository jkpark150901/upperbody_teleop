@echo off
REM Records one teaching-data episode with scripts/mujoco_physics_process.py
REM (frames.jsonl + images/<camera>/*.png, synced to the camera's own render
REM rate -- see teaching_data_generation_rule.md). Drive the robot via
REM run_all.bat's process 1/2 as usual while this window is open; close the
REM viewer window to stop and finalize the recording.
REM
REM Edit DATASET_DIR/TASK below, or just run `record_dataset.bat <dir> <task>`.

set CONDA_ROOT=C:\Users\admin\miniforge3
set ENV_NAME=upperbody_teleop
set PROJECT_ROOT=%~dp0

set DATASET_DIR=recordings\ep001
set TASK=pick up the cloth
if not "%~1"=="" set DATASET_DIR=%~1
if not "%~2"=="" set TASK=%~2

call "%CONDA_ROOT%\Scripts\activate.bat" %ENV_NAME%
cd /d "%PROJECT_ROOT%"
python -m scripts.mujoco_physics_process --port 6100 --record-dataset "%DATASET_DIR%" --task "%TASK%"
