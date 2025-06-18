#!/usr/bin/env python3
"""Basic test script to verify KCVM components"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        from hid.mouse import MouseHID
        print("✓ Mouse HID module")
    except ImportError as e:
        print(f"✗ Mouse HID module: {e}")
        
    try:
        from hid.keyboard import KeyboardHID
        print("✓ Keyboard HID module")
    except ImportError as e:
        print(f"✗ Keyboard HID module: {e}")
        
    try:
        from vision.screen_detector import ScreenDetector
        print("✓ Screen detector module")
    except ImportError as e:
        print(f"✗ Screen detector module: {e}")
        
    try:
        from calibration.calibrator import Calibrator
        print("✓ Calibrator module")
    except ImportError as e:
        print(f"✗ Calibrator module: {e}")
        
    try:
        from tracking.cursor_tracker import CursorTracker
        print("✓ Cursor tracker module")
    except ImportError as e:
        print(f"✗ Cursor tracker module: {e}")
        
    try:
        from video_pipeline import VideoPipeline
        print("✓ Video pipeline module")
    except ImportError as e:
        print(f"✗ Video pipeline module: {e}")
        

def test_config():
    """Test configuration loading"""
    print("\nTesting configuration...")
    
    try:
        import yaml
        with open("config/default.yaml", 'r') as f:
            config = yaml.safe_load(f)
        print("✓ Configuration loaded successfully")
        print(f"  - Camera resolution: {config['camera']['resolution']}")
        print(f"  - Target FPS: {config['camera']['fps']}")
        print(f"  - AI acceleration: {config['performance']['use_ai_acceleration']}")
    except Exception as e:
        print(f"✗ Configuration loading: {e}")
        

def test_hid_devices():
    """Test HID device paths"""
    print("\nTesting HID devices...")
    
    from pathlib import Path
    
    mouse_device = Path("/dev/hidg0")
    keyboard_device = Path("/dev/hidg1")
    
    if mouse_device.exists():
        print("✓ Mouse device found: /dev/hidg0")
    else:
        print("✗ Mouse device not found (run setup_usb_gadget.sh as root)")
        
    if keyboard_device.exists():
        print("✓ Keyboard device found: /dev/hidg1")
    else:
        print("✗ Keyboard device not found (run setup_usb_gadget.sh as root)")
        

if __name__ == "__main__":
    print("KCVM Basic Test Suite")
    print("=" * 50)
    
    test_imports()
    test_config()
    test_hid_devices()
    
    print("\nTest complete!")