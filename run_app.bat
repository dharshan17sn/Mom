@echo off
echo ======================================================================
echo  Starting MA375TGR Multi-Stem Audio Separation & Parametric EQ App
echo ======================================================================
cd /d "%~dp0"
.\.venv\Scripts\streamlit.exe run app.py
pause

