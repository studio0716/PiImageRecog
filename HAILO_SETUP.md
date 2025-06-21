# Hailo-8L AI HAT Setup for KCVM

## What We've Done

1. **Installed Hailo Software Stack**
   ```bash
   sudo apt install hailo-all
   ```
   This installed:
   - HailoRT runtime
   - Hailo TAPPAS Core
   - GStreamer plugins
   - rpicam integration
   - Python bindings

2. **Enabled PCIe Gen 3**
   Added to `/boot/firmware/config.txt`:
   ```
   dtparam=pciex1_gen=3
   ```

3. **Verified Hardware Detection**
   - Hailo-8L detected at `0001:01:00.0`
   - Firmware files present in `/lib/firmware/hailo/`

## Next Steps After Reboot

1. **Check Hailo Status**
   ```bash
   sudo hailortcli fw-control identify
   ```

2. **Verify Kernel Module**
   ```bash
   dmesg | grep -i hailo
   lsmod | grep hailo
   ```

3. **Test with Camera**
   ```bash
   # Basic test
   rpicam-hello
   
   # Object detection with YOLOv6
   rpicam-hello -t 0 --post-process-file /usr/share/rpi-camera-assets/hailo_yolov6_inference.json --lores-width 640 --lores-height 640
   ```

4. **Install Python Bindings**
   ```bash
   pip3 install hailort --break-system-packages
   ```

## Integrating with KCVM Vision System

Once Hailo is working, the AI-powered vision system will automatically use it for:
- Object detection (YOLOv5/v6/v8)
- Text detection (EAST)
- Semantic segmentation
- Real-time inference at 26 TOPS

## Troubleshooting

If firmware still fails to load:
1. Check power supply (27W recommended)
2. Ensure M.2 HAT is properly seated
3. Update firmware:
   ```bash
   sudo rpi-eeprom-update -a
   ```

## Available Models

The Hailo-8L can run these pre-optimized models:
- `hailo_yolov6_inference.json` - General object detection
- `hailo_yolov8_inference.json` - Latest YOLO version
- `hailo_yolox_inference.json` - YOLOX variant
- `hailo_yolov5_personface.json` - Person and face detection
- `hailo_yolov5_segmentation.json` - Instance segmentation
- `hailo_yolov8_pose.json` - Pose estimation

## Performance

With Hailo-8L active:
- 26 TOPS AI acceleration
- Real-time inference at 30+ FPS
- Low power consumption (~5W)
- Offloads AI from CPU/GPU