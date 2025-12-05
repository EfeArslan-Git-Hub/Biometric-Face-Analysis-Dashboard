@echo off
echo Starting Biometric Dashboard...
echo Using local virtual environment...
set DEEPFACE_HOME=%CD%\deepface_home
CALL venv\Scripts\activate
python -m streamlit run app.py
pause
