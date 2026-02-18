#!/bin/bash
# =============================================================================
# AI Avatar - Start Headless Mode
# =============================================================================
# Runs the AI Avatar without a display window
# Useful for AWS EC2 or servers without X11
#
# Usage: ./start_headless.sh [options]
#   --debug        Enable debug mode
# =============================================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

DEBUG=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            DEBUG="--debug"
            shift
            ;;
        *)
            shift
            ;;
    esac
done

# Find the virtual environment
if [ -d ".venv" ]; then
    VENV=".venv"
elif [ -d "venv" ]; then
    VENV="venv"
else
    echo -e "${YELLOW}No virtual environment found. Creating one...${NC}"
    python3 -m venv .venv
    VENV=".venv"
    source $VENV/bin/activate
    pip install -r requirements.txt
fi

# Activate virtual environment
source $VENV/bin/activate

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}   AI Avatar - Headless Mode${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""
echo "Running without display window..."
echo "Press Ctrl+C to stop"
echo ""

# Run in headless mode
python main.py --no-display $DEBUG
