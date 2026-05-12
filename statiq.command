#!/bin/bash
# Doppelklick-Launcher für statiq
# venv liegt unter ~/.local/share/statiq/ (außerhalb iCloud-synced Ordner)
cd "$(dirname "$0")"
VENV="$HOME/.local/share/statiq/.venv"
if [ ! -x "$VENV/bin/python" ]; then
    echo "Setup: venv wird angelegt unter $VENV"
    python3 -m venv "$VENV" || { echo "venv-Anlage fehlgeschlagen"; read -p "Enter zum Schließen..."; exit 1; }
    "$VENV/bin/pip" install --quiet 'PyQt6==6.7.1' scipy numpy matplotlib || { echo "pip install fehlgeschlagen"; read -p "Enter zum Schließen..."; exit 1; }
fi
exec "$VENV/bin/python" main.py
