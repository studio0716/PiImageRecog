#!/usr/bin/env python3
"""Quick model switcher for Hailo detection"""

import sys
import os

MODELS = {
    "yolov8s": {
        "path": "yolov8s.hef",
        "desc": "Small & Fast (30 FPS) - 80 COCO classes"
    },
    "yolov8m": {
        "path": "yolov8m.hef", 
        "desc": "Medium & Accurate (17 FPS) - 80 COCO classes"
    },
    "yolov11s": {
        "path": "yolov11s.hef",
        "desc": "Latest Architecture (25 FPS) - 80 COCO classes"  
    },
    "yolov11n": {
        "path": "yolov11n.hef",
        "desc": "Nano & Very Fast (35+ FPS) - 80 COCO classes"
    }
}

def switch_model(model_name):
    """Switch to a different YOLO model"""
    if model_name not in MODELS:
        print(f"Unknown model: {model_name}")
        print(f"Available models: {', '.join(MODELS.keys())}")
        return
        
    model_info = MODELS[model_name]
    hef_path = f"/home/pi/hailo-rpi5-examples/resources/{model_info['path']}"
    
    # Update the hailo_proper_web.py file
    print(f"Switching to {model_name}: {model_info['desc']}")
    
    # Create sed command to update the model path
    sed_cmd = f"sed -i 's|self.hef_path = .*|self.hef_path = \"{hef_path}\"|' /home/pi/hailo_proper_web.py"
    
    print(f"\nTo switch on the Pi, run:")
    print(f"ssh pi@kcvm.local \"{sed_cmd} && sudo systemctl restart hailo-proper\"")
    print(f"\nThen check: http://kcvm.local:8080")

if __name__ == "__main__":
    print("=== Hailo Model Switcher ===\n")
    
    if len(sys.argv) > 1:
        switch_model(sys.argv[1])
    else:
        print("Available models:")
        for name, info in MODELS.items():
            print(f"  {name}: {info['desc']}")
        print(f"\nUsage: python3 {sys.argv[0]} <model_name>")
        print(f"Example: python3 {sys.argv[0]} yolov11s")