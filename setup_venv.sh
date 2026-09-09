#!/usr/bin/env bash
set -e

VENV_DIR=".venv"

if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment '$VENV_DIR' already exists."
    read -p "Recreate it? [y/N] " answer
    if [[ "$answer" =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_DIR"
    else
        echo "Activating existing venv."
        source "$VENV_DIR/bin/activate"
        echo "Done. Python: $(which python)"
        exit 0
    fi
fi

echo "Creating virtual environment in '$VENV_DIR'..."
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing package (editable, with dev and enrich extras)..."
pip install -e ".[dev,enrich]"

echo ""
echo "Done. To activate the venv run:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Remember to install the scraper's browser fallback:"
echo "  playwright install chromium"
