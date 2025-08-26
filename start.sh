#!/bin/bash
# This script starts the AI Hand-Drawn Map Generator server.

# Get the directory where the script is located to ensure paths are correct
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
VENV_PATH="$SCRIPT_DIR/.venv"

# Check if the virtual environment exists and activate it
if [ -d "$VENV_PATH" ]; then
  echo "Activating virtual environment from: $VENV_PATH"
  source "$VENV_PATH/bin/activate"
else
  echo "Warning: Virtual environment not found at $VENV_PATH."
  echo "Attempting to run with the current environment's packages."
fi

echo ""
echo "Starting AI Hand-Drawn Map Generator server (Fixed MPS Version) on http://127.0.0.1:8002"
echo "Press Ctrl+C to stop the server."

# Run the FIXED MPS version of the application on port 8002
uvicorn main_mps_fixed:app --host 127.0.0.1 --port 8002
