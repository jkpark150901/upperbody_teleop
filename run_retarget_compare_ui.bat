@echo off
REM Open a launcher UI for comparing the same Quest recording on DG-5F and AnyDex hands.

set CONDA_ROOT=C:\Users\admin\miniforge3
set ENV_NAME=upperbody_teleop
set PROJECT_ROOT=%~dp0

call "%CONDA_ROOT%\Scripts\activate.bat" %ENV_NAME%
cd /d "%PROJECT_ROOT%"
python -m scripts.retarget_compare_ui
