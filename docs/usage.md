# KCVM Usage Guide

This guide explains how to use KCVM to control your computer with the Raspberry Pi camera system.

## Initial Setup

### First Connection

1. **Position the Camera**
   - Place the camera 30-50cm from your monitor
   - Ensure the entire screen is visible in the camera view
   - Minimize angle - camera should face screen directly
   - Secure the camera mount to prevent movement

2. **Connect to Target Computer**
   - Use the USB-C data port (right port) on the Pi
   - Connect to any USB port on your target computer
   - The Pi will appear as a generic HID device (no drivers needed)

3. **Power On**
   - Connect power to the left USB-C port
   - Wait for the system to boot (green LED activity)
   - The KCVM service will start automatically

## Calibration Process

### Automatic Calibration

When KCVM starts for the first time or detects a new screen:

1. **Screen Detection Phase**
   - KCVM will detect the screen edges
   - You'll see the mouse cursor move to the corners
   - This maps the camera view to screen coordinates

2. **Cursor Detection Calibration**
   - The system will make small mouse movements
   - It learns to recognize the cursor appearance
   - This process takes about 30 seconds

3. **Save Calibration**
   - Calibration is saved automatically
   - Reused on subsequent connections

### Manual Calibration

If automatic calibration fails:

```bash
# SSH into your Pi
ssh pi@kcvm.local

# Run manual calibration
cd ~/KCVM
python3 src/calibration/manual_calibrate.py
```

Follow on-screen prompts to:
1. Click on four corners of the screen
2. Identify cursor when highlighted
3. Test mouse movements

## Operating Modes

### 1. Mouse Control Mode (Default)

KCVM tracks your screen and allows:
- Precise cursor positioning
- Click actions (left, right, middle)
- Drag and drop operations
- Scroll wheel emulation

Controls:
- **Move**: Camera tracks desired position
- **Click**: Gesture or command based
- **Drag**: Hold and move gestures

### 2. Keyboard Mode

Switch to keyboard mode for text input:
- OCR-based text recognition
- Virtual keyboard overlay
- Gesture-based typing

### 3. Hybrid Mode

Seamlessly switch between mouse and keyboard:
- Context-aware mode switching
- Optimized for productivity

## Control Methods

### Gesture Control (Default)

Use hand gestures in front of camera:
- **Point**: Move cursor
- **Pinch**: Click
- **Spread**: Right-click
- **Swipe**: Scroll

### Voice Control (Optional)

Enable voice commands:
```bash
# Edit config
nano ~/KCVM/config/default.yaml
# Set enable_voice: true
```

Commands:
- "Click" - Left click
- "Right click" - Right click
- "Type [text]" - Keyboard input
- "Scroll up/down" - Scroll

### API Control

Control via network API:
```python
import requests

# Move mouse
requests.post('http://kcvm.local:8080/mouse/move', 
              json={'x': 500, 'y': 300})

# Click
requests.post('http://kcvm.local:8080/mouse/click',
              json={'button': 'left'})

# Type text
requests.post('http://kcvm.local:8080/keyboard/type',
              json={'text': 'Hello World'})
```

## Configuration

### Config File Location

```bash
~/KCVM/config/default.yaml
```

### Key Settings

```yaml
# Camera settings
camera:
  resolution: [1920, 1080]
  fps: 30
  exposure_mode: auto

# Detection settings  
detection:
  screen_threshold: 0.8
  cursor_confidence: 0.9
  
# Performance
performance:
  use_ai_acceleration: true
  target_latency_ms: 50

# Control settings
control:
  mouse_sensitivity: 1.0
  click_delay_ms: 100
  gesture_mode: true
  voice_mode: false
```

## Monitoring and Debugging

### View Logs

```bash
# Real-time logs
journalctl -u kcvm.service -f

# Full logs
journalctl -u kcvm.service --since today
```

### Performance Metrics

```bash
# Check system performance
cd ~/KCVM
python3 src/tools/performance_monitor.py
```

Displays:
- FPS (frames per second)
- Latency (ms)
- CPU/GPU usage
- Detection accuracy

### Debug Mode

Enable debug output:
```bash
# Start in debug mode
sudo systemctl stop kcvm
cd ~/KCVM
python3 src/main.py --debug
```

Shows:
- Camera feed with overlays
- Detection boundaries
- Cursor tracking
- Performance stats

## Troubleshooting

### No Mouse Movement

1. Check USB connection:
   ```bash
   ls /dev/hidg*
   ```

2. Verify calibration:
   ```bash
   cat ~/KCVM/config/calibration.json
   ```

3. Re-run calibration if needed

### Laggy Performance

1. Check AI acceleration:
   ```bash
   hailortcli fw-control identify
   ```

2. Reduce resolution in config
3. Ensure good lighting conditions

### Inaccurate Cursor Tracking

1. Clean camera lens
2. Improve lighting (avoid screen glare)
3. Re-calibrate with stable camera mount
4. Check for screen refresh interference

## Advanced Features

### Multi-Monitor Support

Configure additional screens:
```yaml
screens:
  - id: primary
    position: [0, 0]
    resolution: [1920, 1080]
  - id: secondary  
    position: [1920, 0]
    resolution: [1920, 1080]
```

### Macro Recording

Record and replay actions:
```bash
# Start recording
python3 src/tools/macro_recorder.py --record macro1

# Replay
python3 src/tools/macro_recorder.py --play macro1
```

### Custom Gestures

Add custom gestures in config:
```yaml
custom_gestures:
  - name: "peace_sign"
    action: "screenshot"
  - name: "thumbs_up"
    action: "volume_up"
```

## Security Considerations

1. **USB HID Security**
   - KCVM acts as a trusted HID device
   - No authentication by default
   - Use in trusted environments only

2. **Network API**
   - Disabled by default
   - Enable authentication if using
   - Use HTTPS for remote access

3. **Privacy**
   - Camera only active during use
   - No screen recording by default
   - Processed locally on Pi

## Tips for Best Results

1. **Lighting**: Consistent, indirect lighting works best
2. **Stability**: Mount camera securely to prevent drift
3. **Distance**: Keep consistent distance from screen
4. **Screen**: Matte screens work better than glossy
5. **Performance**: Use wired ethernet for remote API