@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-v1.2.ps1" %*
exit /b %ERRORLEVEL%
