#!/bin/bash
# =============================================================================
# AI Avatar - Cloud VM Quick Setup (Legacy Wrapper)
# =============================================================================
# This script now wraps aws_setup.sh for backward compatibility.
# For full AWS Linux support, use: ./aws_setup.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "AI Avatar - Cloud Setup"
echo "============================================"
echo ""
echo "This script now uses aws_setup.sh for improved AWS Linux support."
echo ""

# Check if aws_setup.sh exists
if [ -f "aws_setup.sh" ]; then
    chmod +x aws_setup.sh
    exec ./aws_setup.sh "$@"
else
    echo "ERROR: aws_setup.sh not found!"
    echo "Please ensure you have the complete repository."
    exit 1
fi
