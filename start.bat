@echo off
setlocal
cd /d "%~dp0"
rem Python 3.10-3.13: older ones can't install some packages (truststore needs 3.10),
rem newer ones may not have torch builds yet. A .venv made with another one starts over.
if not exist .venv goto make
.venv\Scripts\python -c "import sys; sys.exit(not (3, 10) <= sys.version_info[:2] <= (3, 13))" 2>nul && goto install
echo Setting up again with a newer Python...
rmdir /s /q .venv
:make
for %%v in (3.13 3.12 3.11 3.10) do py -%%v -c "" 2>nul && set "PYV=%%v" && goto found
echo Track Splitter needs Python 3.10 to 3.13, and none was found.
echo Install Python 3.13 from https://www.python.org/downloads/ and run this again.
pause
exit /b 1
:found
echo First run: setting up with Python %PYV%, this takes a few minutes...
py -%PYV% -m venv .venv || exit /b 1
:install
rem (Re)install packages on the first run and whenever requirements.txt changes.
.venv\Scripts\python -c "import os, sys; s = '.venv/.installed'; sys.exit(0 if os.path.exists(s) and os.path.getmtime(s) >= os.path.getmtime('requirements.txt') else 1)"
if errorlevel 1 (
    echo Installing packages...
    .venv\Scripts\python -m pip install --upgrade pip -q
    .venv\Scripts\python -m pip install -r requirements.txt || exit /b 1
    type nul > .venv\.installed
)
.venv\Scripts\python app.py
