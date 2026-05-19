@echo off
cd /d %~dp0

:loop
echo [%DATE% %TIME%] Starting Synk bot...
python main.py
echo [%DATE% %TIME%] Bot exited (code %ERRORLEVEL%). Restarting in 15 seconds...
timeout /t 15 /nobreak >nul
goto loop
