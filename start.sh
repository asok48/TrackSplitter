#!/bin/sh
# macOS/Linux launcher. Run with: sh start.sh
cd "$(dirname "$0")" || exit 1
if [ ! -d .venv ]; then
    echo "First run: setting up, this takes a few minutes..."
    python3 -m venv .venv || exit 1
fi
# (Re)install packages on the first run and whenever requirements.txt changes.
if [ ! -f .venv/.installed ] || [ requirements.txt -nt .venv/.installed ]; then
    echo "Installing packages..."
    .venv/bin/python -m pip install --upgrade pip -q
    .venv/bin/python -m pip install -r requirements.txt || exit 1
    touch .venv/.installed
fi
.venv/bin/python app.py
