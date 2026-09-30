@echo off
REM Stops the three windows run_all.bat starts, by their window title
REM (/T also kills each cmd window's child python.exe, not just the shell).

taskkill /FI "WINDOWTITLE eq 1: hand_process*" /T /F
taskkill /FI "WINDOWTITLE eq 2: mocap_process*" /T /F
taskkill /FI "WINDOWTITLE eq 3: mujoco_physics_process*" /T /F
