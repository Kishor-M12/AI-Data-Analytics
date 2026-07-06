@echo off
echo ================================================
echo  RAG Analytics Platform — Starting Services
echo ================================================

REM Terminal 1: FastAPI backend — activate venv then run uvicorn
start "RAG Backend" cmd /k "cd /d %~dp0 && .venv\Scripts\activate && uvicorn main:app --host 0.0.0.0 --port 8000 --reload"

REM Wait for backend to initialize
timeout /t 4 /nobreak >nul

REM Terminal 2: Streamlit frontend — activate venv then run streamlit
start "RAG Frontend" cmd /k "cd /d %~dp0 && .venv\Scripts\activate && streamlit run frontend/app.py --server.port 8501"

echo.
echo  Backend  : http://localhost:8000
echo  API Docs : http://localhost:8000/docs
echo  Frontend : http://localhost:8501
echo.
pause
