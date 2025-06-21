#!/usr/bin/env python3
"""YOLOv11 Hailo Web Detection - Using newer YOLO architecture"""

import os
import sys
import cv2
import numpy as np
import threading
import time
import queue
import json
from flask import Flask, Response, render_template_string
import logging

# Add Hailo examples to path
sys.path.insert(0, '/home/pi/hailo-rpi5-examples')

# Import Hailo libraries
try:
    from hailo_platform import (HEF, VDevice, HailoStreamInterface, ConfigureParams,
                               InputVStreamParams, OutputVStreamParams, FormatType,
                               InferVStreams)
    HAILO_AVAILABLE = True
except ImportError:
    HAILO_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables
frame_queue = queue.Queue(maxsize=10)
detection_queue = queue.Queue(maxsize=10)
stats = {'fps': 0, 'objects': 0, 'detections': [], 'hailo_status': 'Starting...', 'model': 'YOLOv11s'}

# COCO class names (same 80 classes)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

class HailoYOLO11Detector:
    def __init__(self):
        self.running = True
        self.device = None
        self.network_group = None
        self.camera_thread = None
        self.inference_thread = None
        # Using YOLOv11s - newer architecture with better performance
        self.hef_path = '/home/pi/hailo-rpi5-examples/resources/yolov11s.hef'
        
    def start(self):
        """Initialize and start all components"""
        # Initialize Hailo device
        if not self.initialize_hailo():
            stats['hailo_status'] = 'Failed to initialize Hailo'
            logger.error("Hailo initialization failed")
            return
            
        # Start camera thread
        self.camera_thread = threading.Thread(target=self.camera_capture_thread)
        self.camera_thread.daemon = True
        self.camera_thread.start()
        
        # Start inference thread
        self.inference_thread = threading.Thread(target=self.inference_thread_func)
        self.inference_thread.daemon = True
        self.inference_thread.start()
        
        stats['hailo_status'] = 'Running'
        logger.info("Hailo YOLOv11 detector started successfully")
        
    def initialize_hailo(self):
        """Initialize Hailo device and load model"""
        try:
            if not HAILO_AVAILABLE:
                logger.error("Hailo libraries not available")
                return False
                
            # Initialize VDevice
            logger.info("Initializing Hailo VDevice...")
            self.device = VDevice()
            
            # Load HEF file
            logger.info(f"Loading HEF file: {self.hef_path}")
            hef = HEF(self.hef_path)
            
            # Configure network group
            logger.info("Configuring network group...")
            configure_params = ConfigureParams.create_from_hef(hef=hef, interface=HailoStreamInterface.PCIe)
            network_groups = self.device.configure(hef, configure_params)
            self.network_group = network_groups[0]
            self.network_group_params = self.network_group.create_params()
            
            # Create input and output virtual streams params
            self.input_vstream_params = InputVStreamParams.make(self.network_group, format_type=FormatType.FLOAT32)
            self.output_vstream_params = OutputVStreamParams.make(self.network_group, format_type=FormatType.FLOAT32)
            
            # Get stream info
            self.input_vstream_info = hef.get_input_vstream_infos()
            self.output_vstream_info = hef.get_output_vstream_infos()
            
            # Log network info
            logger.info(f"Network group configured successfully")
            logger.info(f"Model: YOLOv11s")
            logger.info(f"Input streams: {len(self.input_vstream_info)}")
            logger.info(f"Output streams: {len(self.output_vstream_info)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Hailo: {e}")
            return False
            
    def camera_capture_thread(self):
        """Capture frames from camera"""
        try:
            from picamera2 import Picamera2
            
            camera = Picamera2()
            config = camera.create_preview_configuration(
                main={"size": (1280, 720), "format": "RGB888"}
            )
            camera.configure(config)
            camera.start()
            
            logger.info("Camera started")
            
            while self.running:
                frame = camera.capture_array()
                frame = cv2.rotate(frame, cv2.ROTATE_180)
                
                # Put frame in queue for inference
                if not frame_queue.full():
                    frame_queue.put(frame)
                    
                time.sleep(0.03)  # ~30 FPS
                
            camera.stop()
            camera.close()
            
        except Exception as e:
            logger.error(f"Camera thread error: {e}")
            
    def inference_thread_func(self):
        """Run inference on frames"""
        try:
            frame_count = 0
            fps_timer = time.time()
            
            # Create inference pipeline
            with InferVStreams(self.network_group, self.input_vstream_params, self.output_vstream_params) as infer_pipeline:
                with self.network_group.activate(self.network_group_params):
                    logger.info("Inference pipeline activated")
                    
                    # Get input shape from stream info
                    input_shape = self.input_vstream_info[0].shape
                    logger.info(f"Input shape for inference: {input_shape}")
                    
                    while self.running:
                        if not frame_queue.empty():
                            frame = frame_queue.get()
                            
                            # Preprocess frame for YOLOv11
                            input_data = self.preprocess_frame(frame, input_shape)
                            
                            # Run inference
                            detections = self.run_inference(input_data, infer_pipeline)
                            
                            # Draw detections on frame
                            annotated_frame = self.draw_detections(frame, detections)
                            
                            # Update stats
                            stats['objects'] = len(detections)
                            stats['detections'] = detections
                            
                            # Put annotated frame for display
                            if not detection_queue.full():
                                detection_queue.put(annotated_frame)
                            
                            # Calculate FPS
                            frame_count += 1
                            if time.time() - fps_timer > 1.0:
                                stats['fps'] = frame_count / (time.time() - fps_timer)
                                frame_count = 0
                                fps_timer = time.time()
                                
                        time.sleep(0.01)
                        
        except Exception as e:
            logger.error(f"Inference thread error: {e}")
            stats['hailo_status'] = f'Error: {str(e)}'
            
    def preprocess_frame(self, frame, input_shape):
        """Preprocess frame for YOLO inference"""
        # YOLOv11 expects 640x640 RGB input
        # input_shape is (H, W, C) for NHWC format
        if len(input_shape) == 3:
            target_size = (input_shape[1], input_shape[0])  # (W, H) for cv2.resize
        else:
            # If shape is 4D (batch, H, W, C), extract H and W
            target_size = (input_shape[2], input_shape[1])
        
        # Resize frame
        resized = cv2.resize(frame, target_size)
        
        # Convert to float32 - YOLOv11 models typically expect [0, 255] range
        normalized = resized.astype(np.float32)
        
        # Add batch dimension for NHWC format
        input_data = np.expand_dims(normalized, axis=0)
            
        return input_data
        
    def run_inference(self, input_data, infer_pipeline):
        """Run inference using Hailo"""
        try:
            # Create input dictionary with stream names from info
            input_dict = {self.input_vstream_info[0].name: input_data}
            
            # Run inference
            raw_outputs = infer_pipeline.infer(input_dict)
            
            # Process outputs
            detections = self.process_yolo_output(raw_outputs)
            
            return detections
            
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return []
            
    def process_yolo_output(self, raw_outputs):
        """Process YOLO model outputs"""
        detections = []
        
        try:
            # YOLOv11 output processing
            # Handle NMS post-processed output
            if isinstance(raw_outputs, dict):
                for output_name, output_data in raw_outputs.items():
                    # Handle NMS post-processed output
                    if 'nms_postprocess' in output_name and isinstance(output_data, list) and len(output_data) > 0:
                        # output_data[0] contains a list of 80 arrays (one per class)
                        if isinstance(output_data[0], list) and len(output_data[0]) == 80:
                            # Process each class
                            for class_id, class_detections in enumerate(output_data[0]):
                                if isinstance(class_detections, np.ndarray) and len(class_detections) > 0:
                                    # Each detection is [x1, y1, x2, y2, score]
                                    for det in class_detections:
                                        if len(det) >= 5:
                                            x1, y1, x2, y2, score = det[:5]
                                            if score > 0.25:  # Confidence threshold
                                                # Check if coordinates are already normalized
                                                if x2 <= 2.0 and y2 <= 2.0:
                                                    bbox = [float(max(0, x1)), float(max(0, y1)), 
                                                           float(min(1, x2)), float(min(1, y2))]
                                                else:
                                                    # Need to normalize by inference size (640x640)
                                                    bbox = [float(x1/640), float(y1/640), float(x2/640), float(y2/640)]
                                                
                                                # Apply same rotation fix as YOLOv8
                                                # -90 degree rotation + horizontal flip
                                                x1_rot = 1.0 - bbox[3]  # new x1 = 1 - old y2
                                                y1_rot = bbox[0]  # new y1 = old x1
                                                x2_rot = 1.0 - bbox[1]  # new x2 = 1 - old y1
                                                y2_rot = bbox[2]  # new y2 = old x2
                                                
                                                # Ensure proper ordering (x1 < x2, y1 < y2)
                                                bbox = [
                                                    float(min(x1_rot, x2_rot)),
                                                    float(min(y1_rot, y2_rot)),
                                                    float(max(x1_rot, x2_rot)),
                                                    float(max(y1_rot, y2_rot))
                                                ]
                                                
                                                # Flip horizontally
                                                bbox = [
                                                    1.0 - bbox[2],  # new x1 = 1 - old x2
                                                    bbox[1],        # y1 stays same
                                                    1.0 - bbox[0],  # new x2 = 1 - old x1
                                                    bbox[3]         # y2 stays same
                                                ]
                                                
                                                detections.append({
                                                    'bbox': bbox,
                                                    'confidence': float(score),
                                                    'class_id': class_id,
                                                    'label': COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else 'unknown'
                                                })
                                    
        except Exception as e:
            logger.error(f"Output processing error: {e}")
            
        return detections[:20]  # Limit to top 20 detections
        
    def draw_detections(self, frame, detections):
        """Draw detection boxes and labels on frame"""
        annotated = frame.copy()
        h, w = frame.shape[:2]
        
        for det in detections:
            # bbox is already in [x1, y1, x2, y2] normalized format
            x1, y1, x2, y2 = det['bbox']
            
            # Convert to pixel coordinates
            x1 = int(x1 * w)
            y1 = int(y1 * h)
            x2 = int(x2 * w)
            y2 = int(y2 * h)
            
            # Get color for class
            color = self.get_class_color(det['label'])
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{det['label']}: {det['confidence']:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Ensure label background doesn't go off screen
            label_y1 = max(0, y1-30)
            label_y2 = max(30, y1)
            
            cv2.rectangle(annotated, (x1, label_y1), (x1+label_size[0]+10, label_y2), color, -1)
            cv2.putText(annotated, label, (x1+5, label_y2-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
        # Add overlay
        self.add_overlay(annotated)
        
        return annotated
        
    def get_class_color(self, class_name):
        """Get color for object class"""
        colors = {
            'person': (255, 0, 255),
            'car': (0, 255, 0),
            'bicycle': (255, 255, 0),
            'motorcycle': (255, 127, 0),
            'bus': (0, 255, 255),
            'truck': (127, 255, 0),
            'cat': (255, 0, 127),
            'dog': (127, 0, 255),
            'tv': (0, 127, 255),
            'laptop': (255, 0, 0),
            'cell phone': (255, 255, 0),
            'keyboard': (255, 255, 255),
            'mouse': (0, 0, 255)
        }
        
        if class_name in colors:
            return colors[class_name]
            
        # Generate color from hash
        hash_val = sum(ord(c) for c in class_name)
        return ((hash_val * 37) % 256, (hash_val * 73) % 256, (hash_val * 109) % 256)
        
    def add_overlay(self, frame):
        """Add stats overlay"""
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        cv2.putText(frame, "Hailo-8L YOLOv11s Detection", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {stats['fps']:.1f}", (20, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.putText(frame, f"Objects: {stats['objects']}", (150, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        cv2.putText(frame, f"Status: {stats['hailo_status']}", (20, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Model: {stats['model']}", (20, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 255), 1)
                   
    def stop(self):
        """Stop detector"""
        self.running = False
        if self.device:
            self.device.release()

# Flask routes
@app.route('/')
def index():
    html = '''<!DOCTYPE html>
<html>
<head>
    <title>Hailo YOLOv11 Detection</title>
    <style>
        body { background: #000; color: white; margin: 0; font-family: Arial; }
        .container { max-width: 1280px; margin: 0 auto; padding: 20px; }
        h1 { text-align: center; color: #4ecdc4; }
        #videoFeed { width: 100%; border-radius: 10px; }
        .stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin: 20px 0; }
        .stat { background: #1a1a1a; padding: 20px; border-radius: 10px; text-align: center; }
        .stat-value { font-size: 2em; color: #4ecdc4; }
        .detections { background: #1a1a1a; padding: 20px; border-radius: 10px; margin: 20px 0; }
        .detection-item { display: inline-block; background: #333; padding: 10px; margin: 5px; border-radius: 5px; }
        .model-info { background: #1a1a1a; padding: 15px; border-radius: 10px; margin: 20px 0; text-align: center; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Hailo-8L YOLOv11 Object Detection</h1>
        <img id="videoFeed" src="/video_feed" alt="Detection Feed">
        
        <div class="stats">
            <div class="stat">
                <div class="stat-value" id="fps">0</div>
                <div>FPS</div>
            </div>
            <div class="stat">
                <div class="stat-value" id="objects">0</div>
                <div>Objects</div>
            </div>
            <div class="stat">
                <div class="stat-value">YOLOv11s</div>
                <div>Model</div>
            </div>
            <div class="stat">
                <div class="stat-value" id="status">Starting...</div>
                <div>Status</div>
            </div>
        </div>
        
        <div class="detections">
            <h3>Detected Objects</h3>
            <div id="detections">Waiting for detections...</div>
        </div>
        
        <div class="model-info">
            <strong>YOLOv11s</strong> - Latest YOLO architecture with improved performance<br>
            22% fewer parameters than YOLOv8m but higher mAP • 80 COCO object classes
        </div>
    </div>
    
    <script>
        setInterval(async () => {
            const res = await fetch('/api/stats');
            const data = await res.json();
            document.getElementById('fps').textContent = data.fps.toFixed(1);
            document.getElementById('objects').textContent = data.objects;
            document.getElementById('status').textContent = data.hailo_status;
            
            if (data.detections.length > 0) {
                document.getElementById('detections').innerHTML = 
                    data.detections.map(d => 
                        `<div class="detection-item">${d.label} (${(d.confidence*100).toFixed(0)}%)</div>`
                    ).join('');
            } else {
                document.getElementById('detections').innerHTML = '<div class="detection-item">No objects detected</div>';
            }
        }, 500);
    </script>
</body>
</html>'''
    return html

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            if not detection_queue.empty():
                frame = detection_queue.get()
                _, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            else:
                time.sleep(0.03)
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/stats')
def api_stats():
    return stats

# Global detector
detector = None

if __name__ == '__main__':
    logger.info("Starting Hailo YOLOv11 Web Detection...")
    
    # Check if running in Hailo environment
    if not os.path.exists('/home/pi/hailo-rpi5-examples'):
        logger.error("Hailo examples not found. Please install hailo-rpi5-examples")
        sys.exit(1)
        
    detector = HailoYOLO11Detector()
    detector.start()
    
    try:
        app.run(host='0.0.0.0', port=8080, debug=False)
    finally:
        if detector:
            detector.stop()