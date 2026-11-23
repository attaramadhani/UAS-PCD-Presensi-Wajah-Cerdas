@echo off
cd /d "%~dp0"
set OPENCV_LOG_LEVEL=ERROR
echo Memulai Aplikasi Presensi Wajah Cerdas PCD...
python -m streamlit run app.py
pause
