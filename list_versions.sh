#!/bin/bash
# List all available Hailo versions

echo "=== Hailo Version Management System ==="
echo ""

ssh -o StrictHostKeyChecking=no pi@kcvm.local << 'EOF'
echo "STABLE VERSIONS:"
echo "----------------"
ls -la ~/hailo_versions/stable/*.py 2>/dev/null | awk '{print $9 " (" $5 " bytes)"}'

echo ""
echo "DEVELOPMENT VERSIONS:"
echo "--------------------"
ls -la ~/hailo_versions/development/*.py 2>/dev/null | awk '{print $9 " (" $5 " bytes)"}'

echo ""
echo "CURRENTLY ACTIVE:"
echo "-----------------"
if [ -L ~/hailo_versions/active ]; then
    ls -la ~/hailo_versions/active | awk '{print "Active symlink -> " $11}'
    ACTIVE_FILE=$(readlink ~/hailo_versions/active)
    if [ -f "$ACTIVE_FILE" ]; then
        echo "Active version: $(basename $ACTIVE_FILE)"
        
        # Check which model it's using
        MODEL=$(grep "self.hef_path.*=.*resources/" "$ACTIVE_FILE" | sed 's/.*resources\///; s/.hef.*//')
        echo "Using model: $MODEL"
    fi
else
    echo "No active version set"
fi

echo ""
echo "VERSION INFO:"
echo "-------------"
if [ -f ~/hailo_versions/VERSION_INFO.txt ]; then
    cat ~/hailo_versions/VERSION_INFO.txt
else
    echo "No version info available"
fi
EOF

echo ""
echo "COMMANDS:"
echo "---------"
echo "Switch version:  ./switch_version.sh <filename>"
echo "Backup current:  ./backup_version.sh <description>"
echo "List versions:   ./list_versions.sh"