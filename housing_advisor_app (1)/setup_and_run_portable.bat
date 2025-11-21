@echo off
setlocal ENABLEDELAYEDEXPANSION


set "SEARCH_ROOT=%~dp0"
IF "%SEARCH_ROOT:~-1%"=="\" set "SEARCH_ROOT=%SEARCH_ROOT:~0,-1%"

set "PROJECT="
set "APP_CAND="

IF EXIST "%SEARCH_ROOT%\app.py" (
  set "PROJECT=%SEARCH_ROOT%"
  set "APP_CAND=%SEARCH_ROOT%\app.py"
)


IF NOT DEFINED PROJECT (
  for /f "delims=" %%F in ('dir /b /s "%SEARCH_ROOT%\app.py" 2^>nul') do (
    if not defined PROJECT (
      set "PROJECT=%%~dpF"
      if "!PROJECT:~-1!"=="\" set "PROJECT=!PROJECT:~0,-1!"
      set "APP_CAND=%%F"
    )
  )
)

if not defined PROJECT (
  echo.
  echo Could not find app.py under:
  echo   %SEARCH_ROOT%
  echo.
  set "USER_PATH="
  set /p USER_PATH=Drag your project folder here and press Enter: 
  if not defined USER_PATH ( echo No path entered. Exiting. & pause & exit /b 1 )

  set "USER_PATH=%USER_PATH:"=%"
  if exist "%USER_PATH%\app.py" (
    set "PROJECT=%USER_PATH%"
    set "APP_CAND=%USER_PATH%\app.py"
  ) else (
    echo app.py not found under "%USER_PATH%". Exiting.
    pause & exit /b 1
  )
)

echo.
echo [FOUND] Project: %PROJECT%
echo [FOUND] app.py : %APP_CAND%

pushd "%PROJECT%"


echo.
echo [1/7] Detecting Python...
set "PYEXE="
where py.exe >nul 2>nul && set "PYEXE=py -3"
if not defined PYEXE where python.exe >nul 2>nul && set "PYEXE=python"
if not defined PYEXE where python3.exe >nul 2>nul && set "PYEXE=python3"
if not defined PYEXE (
  echo ERROR: Python 3.x not found. Please install Python 3.10/3.11/3.12.
  pause & popd & exit /b 1
)
echo Found: %PYEXE%

）
echo.
echo [2/7] Creating venv (if missing)...
set "VENV_DIR=%PROJECT%\venv"
IF NOT EXIST "%VENV_DIR%\Scripts\python.exe" (
  %PYEXE% -m venv "%VENV_DIR%" || (echo ERROR creating venv & pause & popd & exit /b 1)
) ELSE (
  echo venv already exists.
)


echo.
echo [3/7] Activating venv...
call "%VENV_DIR%\Scripts\activate.bat" || (echo ERROR activating venv & pause & popd & exit /b 1)


echo.
echo [4/7] Upgrading pip/setuptools/wheel...
python -m pip install --upgrade pip setuptools wheel

echo.
echo [5/7] Installing requirements...
if exist "%PROJECT%\requirements.txt" (
  pip install --only-binary=:all: geopandas shapely pyproj
  pip install -r requirements.txt
) else (
  echo WARNING: requirements.txt not found, skipping auto install.
)


echo.
echo [6/7] Choosing a free port...
set "PORT=8501"
for /f "tokens=*" %%A in ('netstat -ano ^| findstr /r /c:":8501"') do ( set "PORT=8502" )
set "ADDRESS=127.0.0.1"
echo Using %ADDRESS%:%PORT%

echo.
echo [7/7] Launching Streamlit (auto-open)...

python -m streamlit run app.py --server.address %ADDRESS% --server.port %PORT% --server.headless false

popd
endlocal
