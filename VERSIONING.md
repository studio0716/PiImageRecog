# Hailo Version Management System

This system allows you to maintain multiple versions of the Hailo object detection code, switch between them for demos, and develop new features without breaking the stable version.

## Directory Structure

```
/home/pi/hailo_versions/
├── stable/                 # Stable, demo-ready versions
│   ├── hailo_web_v1_yolov8s_original.py    # Original YOLOv8s (~30 FPS)
│   └── hailo_web_v2_yolov8m_working.py     # Current YOLOv8m (~17 FPS)
├── development/            # Work-in-progress versions
│   └── hailo_yolov11_web.py                 # Testing YOLOv11s
├── scripts/                # Version management scripts
│   ├── backup_version.sh
│   ├── switch_version.sh
│   └── list_versions.sh
├── active -> stable/hailo_web_v2_yolov8m_working.py  # Symlink to current
└── VERSION_INFO.txt        # Version history and notes
```

## Current Versions

### Stable Versions
- **v1 - YOLOv8s Original** (`hailo_web_v1_yolov8s_original.py`)
  - Original implementation with YOLOv8s
  - ~30 FPS performance
  - Basic 80 COCO classes
  
- **v2 - YOLOv8m Current** (`hailo_web_v2_yolov8m_working.py`)
  - Upgraded to YOLOv8m for better accuracy
  - ~17 FPS performance  
  - Fixed bounding box orientation (90° rotation + horizontal flip)
  - **Currently Active**

### Development Versions
- **YOLOv11s** (`hailo_yolov11_web.py`)
  - Testing newer YOLO architecture
  - Expected ~25 FPS with better accuracy than YOLOv8s

## Usage

### Switch Between Versions
```bash
# List available versions
~/hailo_versions/scripts/switch_version.sh

# Switch to a specific version
~/hailo_versions/scripts/switch_version.sh hailo_web_v1_yolov8s_original.py
```

### Backup Current Working Version
```bash
# Save current version with description
~/hailo_versions/scripts/backup_version.sh "yolov8m_improved_accuracy"
```

### List All Versions
```bash
~/hailo_versions/scripts/list_versions.sh
```

### Test Development Version
```bash
# Stop current service
sudo systemctl stop hailo-proper

# Run development version manually
cd ~/hailo_versions/development
python3 hailo_yolov11_web.py

# When done, restart stable service
sudo systemctl start hailo-proper
```

## Adding New Versions

1. Develop and test your new version
2. When stable, copy to the stable directory:
   ```bash
   cp ~/new_hailo_version.py ~/hailo_versions/stable/hailo_web_v3_description.py
   ```
3. Update VERSION_INFO.txt with details
4. Use switch_version.sh to activate it

## Service Configuration

The systemd service (`hailo-proper.service`) runs:
```
/usr/bin/python3 /home/pi/hailo_proper_web.py
```

When you switch versions, the script copies the selected version to this location and restarts the service.

## Notes

- Always test new versions in development before moving to stable
- Use backup_version.sh before making major changes
- The active symlink shows which version would be used if the service restarted
- FPS varies by model: YOLOv8s (~30), YOLOv8m (~17), YOLOv11s (~25)
- All current models use 80 COCO classes