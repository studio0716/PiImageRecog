#!/bin/bash
# Backup current Hailo version with timestamp

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <version_description>"
    echo "Example: $0 'yolov8m_improved_accuracy'"
    exit 1
fi

DESCRIPTION=$1
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
FILENAME="hailo_web_${TIMESTAMP}_${DESCRIPTION}.py"

echo "Backing up current version..."
ssh pi@kcvm.local << EOF
if [ -f ~/hailo_proper_web.py ]; then
    cp ~/hailo_proper_web.py ~/hailo_versions/stable/${FILENAME}
    echo "Backed up to: ~/hailo_versions/stable/${FILENAME}"
    
    # Update version info
    echo "" >> ~/hailo_versions/VERSION_INFO.txt
    echo "${TIMESTAMP} - ${DESCRIPTION} (${FILENAME})" >> ~/hailo_versions/VERSION_INFO.txt
    echo "  - Backup of working version" >> ~/hailo_versions/VERSION_INFO.txt
    echo "  - Status: Stable" >> ~/hailo_versions/VERSION_INFO.txt
else
    echo "Error: ~/hailo_proper_web.py not found"
    exit 1
fi
EOF