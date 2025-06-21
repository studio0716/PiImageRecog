#!/bin/bash
# Setup versioning system for Hailo demos on Pi

echo "=== Setting up Hailo Versioning System ==="

# Create directory structure
echo "Creating version directories..."
ssh pi@kcvm.local "mkdir -p ~/hailo_versions/stable ~/hailo_versions/development ~/hailo_versions/scripts"

# Save current working version
echo "Backing up current working version..."
ssh pi@kcvm.local "cp ~/hailo_proper_web.py ~/hailo_versions/stable/hailo_web_v2_yolov8m_working.py"

# Also save the previous YOLOv8s version if we have it
ssh pi@kcvm.local "if [ -f ~/kcvm_hailo_web.py ]; then cp ~/kcvm_hailo_web.py ~/hailo_versions/stable/hailo_web_v1_yolov8s_original.py; fi"

# Create symlink to active version
echo "Creating active version symlink..."
ssh pi@kcvm.local "ln -sf ~/hailo_versions/stable/hailo_web_v2_yolov8m_working.py ~/hailo_versions/active"

# Create version info file
echo "Creating version info file..."
ssh pi@kcvm.local 'cat > ~/hailo_versions/VERSION_INFO.txt << "VERSION_EOF"
=== Hailo Version History ===

v1 - YOLOv8s Original (hailo_web_v1_yolov8s_original.py)
  - Original implementation with YOLOv8s
  - ~30 FPS performance
  - Basic 80 COCO classes
  - Status: Stable

v2 - YOLOv8m Current (hailo_web_v2_yolov8m_working.py) 
  - Upgraded to YOLOv8m for better accuracy
  - ~17 FPS performance
  - Same 80 COCO classes, better detection
  - Fixed bounding box orientation (90° rotation + horizontal flip)
  - Status: Active/Stable

Development:
- YOLOv11s - Testing newer architecture
- Objects365 - Planning for 365 object classes
VERSION_EOF'

# List the created structure
echo "Version system created successfully!"
ssh pi@kcvm.local "ls -la ~/hailo_versions/"