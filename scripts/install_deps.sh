#!/bin/bash

# KCVM Dependency Installation Script

set -e

echo "Installing KCVM dependencies..."

# Check if we're in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Warning: Not in a virtual environment!"
    echo "It's recommended to run this in a Python virtual environment"
    echo "Create one with: python3 -m venv venv && source venv/bin/activate"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Update pip
echo "Updating pip..."
pip install --upgrade pip

# Core dependencies
echo "Installing core Python packages..."
pip install numpy==1.24.3
pip install opencv-python==4.8.1.78
pip install picamera2==0.3.12

# Computer vision and AI packages
echo "Installing computer vision packages..."
pip install Pillow==10.1.0
pip install scikit-image==0.22.0
pip install imutils==0.5.4

# HID and USB packages
echo "Installing HID packages..."
pip install pyusb==1.2.1
pip install evdev==1.6.1

# Async and performance packages
echo "Installing async and performance packages..."
pip install asyncio==3.4.3
pip install uvloop==0.19.0
pip install aiofiles==23.2.1

# Configuration and utilities
echo "Installing utility packages..."
pip install PyYAML==6.0.1
pip install click==8.1.7
pip install python-dotenv==1.0.0

# Web API (optional)
echo "Installing web API packages..."
pip install fastapi==0.104.1
pip install uvicorn==0.24.0
pip install websockets==12.0

# Testing packages
echo "Installing testing packages..."
pip install pytest==7.4.3
pip install pytest-asyncio==0.21.1
pip install pytest-cov==4.1.0

# Development packages
echo "Installing development packages..."
pip install black==23.11.0
pip install flake8==6.1.0
pip install mypy==1.7.1

# Hailo Runtime Python bindings (if available)
echo "Checking for Hailo Python bindings..."
if command -v hailortcli &> /dev/null; then
    echo "Installing Hailo Python bindings..."
    pip install hailort
else
    echo "Hailo toolkit not found. Install it separately for AI acceleration."
fi

# Create requirements.txt for future use
echo "Creating requirements.txt..."
pip freeze > requirements.txt

echo ""
echo "Dependency installation complete!"
echo ""
echo "Note: Some system packages may need to be installed with apt:"
echo "  sudo apt install python3-libcamera python3-kms++"
echo "  sudo apt install libopencv-dev"
echo "  sudo apt install libusb-1.0-0-dev"
echo ""
echo "For AI acceleration, install Hailo toolkit from:"
echo "  https://hailo.ai/developer-zone/"