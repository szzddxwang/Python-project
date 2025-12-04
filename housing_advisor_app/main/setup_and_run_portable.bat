@echo off
setlocal ENABLEDELAYEDEXPANSION

:: =======================
:: 0. Basic path setup
:: =======================
set "SEARCH_ROOT=%~dp0"
IF "%SEARCH_ROOT:~-1%"=="\" set "SEARCH_ROOT=%SEARCH_ROOT:~0,-1%"

set "PROJECT="
set "APP_CAND="

:: First, try to find app.py in the current folder
IF EXIST "%SEARCH_ROOT%\app.py" (
  set "PROJECT=%SEARCH_ROOT%"
  set "APP_CAND=%SEARCH_ROOT%\app.py"
)

:: If not found, recursively search for the first app.py under SEARCH_ROOT
IF NOT DEFINED PROJECT (
  for /f "delims=" %%F in ('dir /b /s "%SEARCH_ROOT%\app.py" 2^>nul') do (
    if not defined PROJECT (
      set "PROJECT=%%~dpF"
      if "!PROJECT:~-1!"=="\" set "PROJECT=!PROJECT:~0,-1!"
      set "APP_CAND=%%F"
    )
  )
)

:: If still not found, ask user to drag project folder into the console
if not defined PROJECT (
  echo.
  echo Could not find app.py under:
  echo   %SEARCH_ROOT%
  echo.
  set "USER_PATH="
  set /p USER_PATH=Drag your project folder here and press Enter: 
  if not defined USER_PATH (
    echo No path entered. Exiting.
    pause
    exit /b 1
  )

  set "USER_PATH=%USER_PATH:"=%"
  if exist "%USER_PATH%\app.py" (
    set "PROJECT=%USER_PATH%"
    set "APP_CAND=%USER_PATH%\app.py"
  ) else (
    echo app.py not found under "%USER_PATH%". Exiting.
    pause
    exit /b 1
  )
)

echo.
echo [FOUND] Project: %PROJECT%
echo [FOUND] app.py : %APP_CAND%

pushd "%PROJECT%"

:: =======================
:: 1. Detect / install Python (use 3.13, >=3.12.0)
:: =======================
echo.
echo [1/7] Detecting Python (use 3.13, >=3.12.0)...

set "PYEXE="

:: 1) Prefer an existing py -3.13 if available
where py.exe >nul 2>nul
if !errorlevel! EQU 0 (
  py -3.13 -c "import sys; print(sys.version)" >nul 2>nul
  if !errorlevel! EQU 0 (
    set "PYEXE=py -3.13"
  )
)

if defined PYEXE (
  echo Found existing Python 3.13: %PYEXE%
) else (
  echo.
  echo No Python 3.13 found. Trying to download and install Python 3.13.7 ^(64-bit^)...
  echo This satisfies the requirement "Python 3.12.0 or later".
  echo.

  set "PY_DL_URL=https://www.python.org/ftp/python/3.13.7/python-3.13.7-amd64.exe"
  set "PY_DL_EXE=%TEMP%\python-3.13.7-amd64.exe"

  :: Use curl if available
  where curl.exe >nul 2>nul
  if !errorlevel! EQU 0 (
    echo Using curl to download installer...
    curl -L -o "%PY_DL_EXE%" "%PY_DL_URL%"
  ) else (
    :: Fallback: PowerShell
    echo curl not found. Trying PowerShell to download installer...
    powershell -Command "try {Invoke-WebRequest -Uri '%PY_DL_URL%' -OutFile '%PY_DL_EXE%' -UseBasicParsing} catch {exit 1}"
  )

  if not exist "%PY_DL_EXE%" (
    echo.
    echo ERROR: Failed to download Python 3.13.7 installer.
    echo Please manually install Python 3.13.7 ^(64-bit^) from:
    echo   https://www.python.org/downloads/release/python-3137/
    echo and then re-run this script.
    pause
    popd
    exit /b 1
  )

  echo.
  echo Running Python 3.13.7 installer ^(this may take a few minutes^)...
  echo If a window pops up, please check "Add python.exe to PATH" and finish the setup.
  echo.

  "%PY_DL_EXE%" /quiet InstallAllUsers=0 PrependPath=1 Include_launcher=1

  :: Re-detect py -3.13 after installation
  set "PYEXE="
  py -3.13 -c "import sys; print(sys.version)" >nul 2>nul
  if !errorlevel! EQU 0 (
    set "PYEXE=py -3.13"
  )

  if not defined PYEXE (
    echo.
    echo ERROR: Python 3.13 still not found after installation attempt.
    echo Please install Python 3.13.7 manually and re-run this script.
    pause
    popd
    exit /b 1
  )

  echo.
  echo Python 3.13 detected successfully: %PYEXE%
)

:: =======================
:: 2. Create venv
:: =======================
echo.
echo [2/7] Creating venv (if missing)...
set "VENV_DIR=%PROJECT%\venv"
IF NOT EXIST "%VENV_DIR%\Scripts\python.exe" (
  %PYEXE% -m venv "%VENV_DIR%"
  if !errorlevel! NEQ 0 (
    echo ERROR creating venv.
    pause
    popd
    exit /b 1
  )
) ELSE (
  echo venv already exists.
)

:: =======================
:: 3. Activate venv
:: =======================
echo.
echo [3/7] Activating venv...
call "%VENV_DIR%\Scripts\activate.bat"
if !errorlevel! NEQ 0 (
  echo ERROR activating venv.
  pause
  popd
  exit /b 1
)

:: =======================
:: 4. Upgrade pip / setuptools / wheel
:: =======================
echo.
echo [4/7] Upgrading pip/setuptools/wheel...
python -m pip install --upgrade pip setuptools wheel

:: =======================
:: 5. Install dependencies
:: =======================
echo.
echo [5/7] Installing requirements...
if exist "%PROJECT%\requirements.txt" (
  echo Installing from requirements.txt...
  python -m pip install -r "%PROJECT%\requirements.txt"
) else (
  echo WARNING: requirements.txt not found, skipping auto install.
)

:: =======================
:: 6. Choose a free port
:: =======================
echo.
echo [6/7] Choosing a free port...
set "PORT=8501"
for /f "tokens=*" %%A in ('netstat -ano ^| findstr /r /c:":8501"') do (
  set "PORT=8502"
)
set "ADDRESS=127.0.0.1"
echo Using %ADDRESS%:%PORT%

:: =======================
:: 7. Launch Streamlit
:: =======================
echo.
echo [7/7] Launching Streamlit (auto-open)...
python -m streamlit run "%APP_CAND%" --server.address %ADDRESS% --server.port %PORT% --server.headless false

echo.
echo Script finished. Press any key to close this window.
pause

popd
endlocal
