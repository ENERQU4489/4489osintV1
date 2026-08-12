@echo off
title 4489 OSINT Tool v1 - AI Geolocation Engine
cd /d "%~dp0"
if exist venv\Scripts\activate.bat call venv\Scripts\activate.bat
if "%1"=="" (
    python osint4489.py
) else (
    python osint4489.py %*
)
pause
