@echo off
setlocal

set "TOOL_DIR=%~dp0"
set "PYTHON_EXE=D:\anaconda3\python.exe"

if exist "%PYTHON_EXE%" (
    set "RUNNER=%PYTHON_EXE%"
) else (
    set "RUNNER=python"
)

start "" http://localhost:8501
"%RUNNER%" -m streamlit run "%TOOL_DIR%app.py" --server.headless=true --browser.gatherUsageStats=false

endlocal
