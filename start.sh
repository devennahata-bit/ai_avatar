#!/bin/bash
# =============================================================================
# AI Avatar - Start Web Server
# =============================================================================
# Starts the AI Avatar web server on port 8080
# Open http://<your-ip>:8080 in your browser
#
# Usage: ./start.sh [options]
#   --port PORT    Port to run on (default: 8080)
#   --debug        Enable debug mode
# =============================================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Default values
PORT=${PORT:-8080}
HOST="0.0.0.0"
DEBUG=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
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
echo -e "${GREEN}   AI Avatar - Web Server${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""
echo -e "Starting server on ${GREEN}http://$HOST:$PORT${NC}"
echo ""
echo "Open this URL in your browser:"
echo -e "  ${GREEN}http://localhost:$PORT${NC}"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Run the web server
python web_server.py --host $HOST --port $PORT $DEBUG
