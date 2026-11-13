@echo off
rem Double-click to start the H2 Solve UI. Runs start.ps1 without changing PowerShell's script policy.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1"
pause
