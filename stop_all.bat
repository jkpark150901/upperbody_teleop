@echo off
REM Stops windows launched by run_all.bat and run_ok_sign_test.bat.
REM /T also kills each cmd window's child python.exe, not just the shell.

taskkill /FI "WINDOWTITLE eq 1: hand_process*" /T /F >nul 2>nul
taskkill /FI "WINDOWTITLE eq 2: mocap_process*" /T /F >nul 2>nul
taskkill /FI "WINDOWTITLE eq 3: mujoco_physics_process*" /T /F >nul 2>nul
taskkill /FI "WINDOWTITLE eq 1: mujoco_physics_process*" /T /F >nul 2>nul
taskkill /FI "WINDOWTITLE eq 2: ok_sign_sender*" /T /F >nul 2>nul
taskkill /FI "WINDOWTITLE eq 1: mujoco_physics_process quest replay*" /T /F >nul 2>nul
taskkill /FI "WINDOWTITLE eq 2: quest_skeleton_replay_sender*" /T /F >nul 2>nul
taskkill /FI "WINDOWTITLE eq 1: mujoco_physics_process pinch ik*" /T /F >nul 2>nul
taskkill /FI "WINDOWTITLE eq 2: pinch_ik_sender*" /T /F >nul 2>nul

echo Stopped upperbody_teleop windows.
