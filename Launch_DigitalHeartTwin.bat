@echo off
setlocal enabledelayedexpansion
title Digital Heart Twin - Launcher
cd /d "%~dp0"
color 0B

echo(
echo ============================================================
echo    Digital Heart Twin  -  Launcher
echo ============================================================
echo(

REM ----- 1. Sanity checks -------------------------------------------------
where python >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Python was not found on PATH.
  echo         Install Python 3, or add it to PATH, then try again.
  echo(
  pause & exit /b 1
)
if not exist "serve_recorder.py" (
  echo [ERROR] serve_recorder.py not found in:
  echo         %cd%
  echo         Put this .bat in the mscardio_project folder and re-run.
  echo(
  pause & exit /b 1
)
if not exist "recorder\index.html" (
  echo [ERROR] recorder\index.html not found - the app is missing.
  echo(
  pause & exit /b 1
)

REM ----- 2. Detect this PC's Wi-Fi / LAN IPv4 (checked BEFORE launch) ------
echo [1/4] Detecting this PC's Wi-Fi / LAN IP address...
set "LANIP="
for /f "usebackq delims=" %%i in (`python -c "import socket;s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);s.connect(('8.8.8.8',80));print(s.getsockname()[0]);s.close()" 2^>nul`) do set "LANIP=%%i"
if not defined LANIP set "LANIP=127.0.0.1"
echo        Wi-Fi / LAN IP : !LANIP!
echo        Phone URL      : https://!LANIP!:8443
echo        This-PC URL    : https://localhost:8443
echo(

REM ----- 3. Ensure the firewall lets phones reach port 8443 ---------------
echo [2/4] Checking Windows Firewall for port 8443 (needed for phone access)...
netsh advfirewall firewall show rule name=Digital Heart TwinRecorder8443 >nul 2>&1
if errorlevel 1 (
  echo        Firewall rule not found.
  choice /C YN /N /M "        Open port 8443 now? A Windows admin prompt will appear [Y/N]: "
  if !errorlevel!==1 (
    echo        Requesting administrator approval - click YES on the prompt...
    powershell -NoProfile -Command "Start-Process -Verb RunAs -FilePath netsh -ArgumentList 'advfirewall','firewall','add','rule','name=Digital Heart TwinRecorder8443','dir=in','action=allow','protocol=TCP','localport=8443'"
    echo        Done ^(if you approved, phones on your Wi-Fi can now connect^).
  ) else (
    echo        Skipped. Local ^(this-PC^) access still works; phone access will not
    echo        until the port is opened.
  )
) else (
  echo        OK - port 8443 is already allowed.
)
echo(

REM ----- 4. Start the server (only if it is not already running) ----------
echo [3/4] Starting the recorder server...
netstat -ano | findstr ":8443" | findstr "LISTENING" >nul 2>&1
if errorlevel 1 (
  start "Digital Heart Twin Server (keep open)" cmd /k python serve_recorder.py --port 8443
  echo        Server starting in its own window ^(keep that window open^).
  powershell -NoProfile -Command "Start-Sleep -Seconds 3" >nul 2>&1
) else (
  echo        A server is already running on port 8443 - reusing it.
)
echo(

REM ----- 5. Open the app on this PC ---------------------------------------
echo [4/4] Opening the app in your browser...
start "" "https://localhost:8443"
echo(

echo ============================================================
echo    READY
echo ------------------------------------------------------------
echo    On THIS PC   : https://localhost:8443
echo    On your PHONE: https://!LANIP!:8443   ^(same Wi-Fi^)
echo(
echo    First time on either device you'll see a certificate
echo    warning - tap/click  Advanced  -^>  Proceed / Continue.
echo(
echo    To STOP: close the black "Digital Heart Twin Server" window.
echo ============================================================
echo(
pause
endlocal
