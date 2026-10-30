@echo off
setlocal enabledelayedexpansion
title Digital Heart Twin - Launcher
cd /d "%~dp0"
color 0B

echo(
echo ============================================================
echo    Digital Heart Twin  -  Launcher (Port 9443)
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
echo        Phone URL      : https://!LANIP!:9443
echo        This-PC URL    : https://localhost:9443
echo(

REM ----- 3. Ensure the firewall lets phones reach port 9443 ---------------
echo [2/4] Checking Windows Firewall for port 9443 (needed for phone access)...
netsh advfirewall firewall show rule name=DigitalHeartTwinRecorder9443 >nul 2>&1
if errorlevel 1 (
  echo        Firewall rule for port 9443 not found.
  choice /C YN /N /M "        Open port 9443 now? A Windows admin prompt will appear [Y/N]: "
  if !errorlevel!==1 (
    echo        Requesting administrator approval - click YES on the prompt...
    powershell -NoProfile -Command "Start-Process -Verb RunAs -FilePath netsh -ArgumentList 'advfirewall','firewall','add','rule','name=DigitalHeartTwinRecorder9443','dir=in','action=allow','protocol=TCP','localport=9443'"
    echo        Done ^(if you approved, phones on your Wi-Fi can now connect^).
  ) else (
    echo        Skipped. Local ^(this-PC^) access still works; phone access will not
    echo        until the port is opened.
  )
) else (
  echo        OK - port 9443 is already allowed.
)
echo(

REM ----- 4. Start the server (stop old process on 8443 and 9443) -----------
echo [3/4] Starting the recorder server on port 9443...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":9443" ^| findstr "LISTENING"') do (
  echo        Stopping previous server on port 9443 (PID %%a)...
  taskkill /F /PID %%a >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8443" ^| findstr "LISTENING"') do (
  echo        Stopping old server on port 8443 (PID %%a)...
  taskkill /F /PID %%a >nul 2>&1
)
start "Digital Heart Twin Server (keep open)" cmd /k python serve_recorder.py --port 9443
echo        Server starting in its own window...
powershell -NoProfile -Command "Start-Sleep -Seconds 2" >nul 2>&1
echo(

REM ----- 5. Open the app on this PC ---------------------------------------
echo [4/4] Opening the app in your browser...
start "" "https://localhost:9443"
echo(

echo ============================================================
echo    READY - SERVER RUNNING ON PORT 9443
echo ------------------------------------------------------------
echo    On THIS PC    : https://localhost:9443
echo    On your IPHONE : https://!LANIP!:9443   (MUST include https://)
echo                     Or: https://!LANIP!:9443/app
echo(
echo    FIRST TIME ON IPHONE SAFARI:
echo    1. You will see "This Connection Is Not Private"
echo    2. Tap "Show Details" (at the bottom)
echo    3. Tap "visit this website" (then confirm "Visit Website")
echo(
echo    To STOP: close the black "Digital Heart Twin Server" window.
echo ============================================================
echo(
pause
endlocal
