@echo off

cd /d "%~dp0.."

echo ==========================================
echo          TalentMatch AI Demo
echo ==========================================
echo.
echo Starting FastAPI...
echo.

start "TalentMatch FastAPI" cmd /k "python -m uvicorn app.10_application.api:app --reload"

timeout /t 3 /nobreak >nul

echo Starting Streamlit...
echo.

start "TalentMatch Streamlit" cmd /k "streamlit run app\10_application\app.py --server.headless true"

timeout /t 5 /nobreak >nul

start "" http://localhost:8501

exit