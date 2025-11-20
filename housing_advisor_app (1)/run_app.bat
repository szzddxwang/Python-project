@echo off
setlocal
REM === Project root (contains app.py) ===
set "PROJECT=C:\Users\szzddx\Desktop\housing_advisor_app (1)\housing_advisor_app"

REM (Optional) Activate venv if it exists
IF EXIST "%PROJECT%\venv\Scripts\activate.bat" call "%PROJECT%\venv\Scripts\activate.bat"

pushd "%PROJECT%"
REM Launch Streamlit (change port with --server.port 8502 if needed)
python -m streamlit run app.py
popd

endlocal
