#!/bin/bash
# Switch between Hailo versions

if [ "$#" -eq 0 ]; then
    echo "=== Available Hailo Versions ==="
    ssh pi@kcvm.local 'ls -1 ~/hailo_versions/stable/*.py | xargs -n1 basename'
    echo ""
    echo "Usage: $0 <version_file>"
    echo "Example: $0 hailo_web_v2_yolov8m_working.py"
    exit 0
fi

VERSION_FILE=$1

echo "Switching to version: $VERSION_FILE"
ssh pi@kcvm.local << EOF
if [ -f ~/hailo_versions/stable/${VERSION_FILE} ]; then
    # Update symlink
    ln -sf ~/hailo_versions/stable/${VERSION_FILE} ~/hailo_versions/active
    
    # Copy to working location
    cp ~/hailo_versions/stable/${VERSION_FILE} ~/hailo_proper_web.py
    
    # Restart service
    sudo systemctl restart hailo-proper
    
    echo "Switched to: ${VERSION_FILE}"
    echo "Service restarted"
    echo "Check: http://kcvm.local:8080"
else
    echo "Error: Version file not found: ${VERSION_FILE}"
    echo "Available versions:"
    ls -1 ~/hailo_versions/stable/*.py | xargs -n1 basename
    exit 1
fi
EOF