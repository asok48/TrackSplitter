@echo off
cd d %~dp0
if not exist .venv (
    echo First run setting up, this takes a few minutes...
    py -3.13 -m venv .venv  exit b 1
)
rem (Re)install packages on the first run and whenever requirements.txt changes.
.venvScriptspython -c import os, sys; s = '.venv.installed'; sys.exit(0 if os.path.exists(s) and os.path.getmtime(s) = os.path.getmtime('requirements.txt') else 1)
if errorlevel 1 (
    echo Installing packages...
    .venvScriptspython -m pip install --upgrade pip -q
    .venvScriptspython -m pip install -r requirements.txt  exit b 1
    type nul  .venv.installed
)
.venvScriptspython app.py
