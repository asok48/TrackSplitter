#!/bin/sh
# macOS/Linux launcher. Run with: sh start.sh
cd "$(dirname "$0")" || exit 1

# Python 3.10-3.13: older ones can't install some packages (truststore needs 3.10;
# Apple's built-in python3 is 3.9), newer ones may not have torch builds yet.
good() { "$1" -c 'import sys; sys.exit(not (3, 10) <= sys.version_info[:2] <= (3, 13))' 2>/dev/null; }

# A .venv made with an unsuitable Python (e.g. Apple's 3.9) can't be fixed: start over.
if [ -d .venv ] && ! good .venv/bin/python; then
    echo "Setting up again with a newer Python..."
    rm -rf .venv
fi
if [ ! -d .venv ]; then
    PYTHON=
    for p in python3.13 python3.12 python3.11 python3.10 python3 \
             /Library/Frameworks/Python.framework/Versions/3.1[0-3]/bin/python3 \
             /opt/homebrew/bin/python3 /usr/local/bin/python3; do
        if command -v "$p" >/dev/null 2>&1 && good "$p"; then PYTHON=$p; break; fi
    done
    if [ -z "$PYTHON" ]; then
        echo "Track Splitter needs Python 3.10 to 3.13, and none was found (a Mac's built-in python3 is 3.9)."
        echo "Install Python 3.13 from https://www.python.org/downloads/ and run this again."
        exit 1
    fi
    echo "First run: setting up with $("$PYTHON" --version), this takes a few minutes..."
    "$PYTHON" -m venv .venv || exit 1
fi
# (Re)install packages on the first run and whenever requirements.txt changes.
if [ ! -f .venv/.installed ] || [ requirements.txt -nt .venv/.installed ]; then
    echo "Installing packages..."
    .venv/bin/python -m pip install --upgrade pip -q
    .venv/bin/python -m pip install -r requirements.txt || exit 1
    touch .venv/.installed
fi
.venv/bin/python app.py
