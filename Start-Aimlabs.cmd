@echo off
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start-aimlabs.ps1" %*
exit /b %errorlevel%
