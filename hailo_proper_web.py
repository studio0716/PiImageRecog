#!/usr/bin/env python3
"""Proper Hailo YOLO Web Detection using actual Hailo APIs"""

import os
import sys
import cv2
import numpy as np
import threading
import time
import queue
import json
import subprocess
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

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables
frame_queue = queue.Queue(maxsize=10)
detection_queue = queue.Queue(maxsize=10)
stats = {'fps': 0, 'objects': 0, 'detections': [], 'hailo_status': 'Starting...'}

# COCO class names
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

class HailoYOLODetector:
    def __init__(self):
        self.running = True
        self.device = None
        self.network_group = None
        self.camera_thread = None
        self.inference_thread = None
        self.hef_path = '/home/pi/hailo-rpi5-examples/resources/yolov8s.hef'
        # For aspect ratio correction
        self.last_scale = 1.0
        self.last_x_offset = 0
        self.last_y_offset = 0
        
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
        logger.info("Hailo YOLO detector started successfully")
        
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
                            
                            # Preprocess frame for YOLOv8
                            input_data = self.preprocess_frame(frame, input_shape)
                            
                            # Run inference
                            logger.debug(f"Running inference with input shape: {input_data.shape}")
                            detections = self.run_inference(input_data, infer_pipeline)
                            logger.debug(f"Got {len(detections)} detections")
                            
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
        # YOLOv8 expects 640x640 RGB input
        # input_shape is (H, W, C) for NHWC format
        if len(input_shape) == 3:
            target_size = (input_shape[1], input_shape[0])  # (W, H) for cv2.resize
        else:
            # If shape is 4D (batch, H, W, C), extract H and W
            target_size = (input_shape[2], input_shape[1])
        
        # Resize frame
        resized = cv2.resize(frame, target_size)
        logger.debug(f"Resized from {frame.shape} to {resized.shape} for inference")
        
        # Convert to float32 - check if normalization is needed
        # Some models expect [0, 255] range, others [0, 1]
        normalized = resized.astype(np.float32)
        # Try without normalization first as some Hailo models expect [0, 255]
        # normalized = resized.astype(np.float32) / 255.0
        
        # Add batch dimension for NHWC format
        input_data = np.expand_dims(normalized, axis=0)
            
        return input_data
        
    def run_inference(self, input_data, infer_pipeline):
        """Run inference using Hailo"""
        try:
            # Create input dictionary with stream names from info
            input_dict = {self.input_vstream_info[0].name: input_data}
            
            # Run inference
            logger.debug(f"Input dict keys: {list(input_dict.keys())}")
            raw_outputs = infer_pipeline.infer(input_dict)
            logger.info(f"Inference complete, output type: {type(raw_outputs)}")
            
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
            # YOLOv8 output processing
            logger.debug(f"Processing output type: {type(raw_outputs)}")
            
            # Handle dict output format from Hailo
            if isinstance(raw_outputs, dict):
                logger.info(f"Output dict keys: {list(raw_outputs.keys())}")
                # Get the first output tensor
                for output_name, output_data in raw_outputs.items():
                    logger.info(f"Processing output '{output_name}' type: {type(output_data)}")
                    if hasattr(output_data, 'shape'):
                        logger.info(f"Output shape: {output_data.shape}")
                    elif isinstance(output_data, list):
                        logger.info(f"Output is list with {len(output_data)} items")
                        if len(output_data) > 0:
                            logger.info(f"First item type: {type(output_data[0])}")
                            if isinstance(output_data[0], list):
                                logger.info(f"Nested list with {len(output_data[0])} items")
                                if len(output_data[0]) > 0:
                                    logger.info(f"First nested item: {output_data[0][0]}")
                                    logger.info(f"First nested item type: {type(output_data[0][0])}")
                                    if hasattr(output_data[0][0], '__dict__'):
                                        logger.info(f"Attributes: {dir(output_data[0][0])}")
                            elif hasattr(output_data[0], '__dict__'):
                                logger.info(f"First item attributes: {output_data[0].__dict__}")
                    else:
                        logger.info(f"Output data: {output_data}")
                    
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
                                            if score > 0.05:  # Very low threshold for testing
                                                # Log raw detection coordinates
                                                logger.debug(f"Raw detection: x1={x1}, y1={y1}, x2={x2}, y2={y2}, class={COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else 'unknown'}")
                                                
                                                # Check if coordinates are already normalized
                                                # Values close to 1.0 indicate normalized coordinates
                                                if x2 <= 2.0 and y2 <= 2.0:
                                                    # Already normalized (allowing for slight overflow)
                                                    bbox = [float(max(0, x1)), float(max(0, y1)), 
                                                           float(min(1, x2)), float(min(1, y2))]
                                                else:
                                                    # Need to normalize by inference size (640x640)
                                                    bbox = [float(x1/640), float(y1/640), float(x2/640), float(y2/640)]
                                                
                                                # Log original bbox for debugging
                                                logger.debug(f"Original bbox: [{bbox[0]:.3f}, {bbox[1]:.3f}, {bbox[2]:.3f}, {bbox[3]:.3f}] for {COCO_CLASSES[class_id]}")
                                                
                                                # The boxes appear to be 90 degrees rotated
                                                # Try -90 degree rotation: (x,y) -> (1-y, x)
                                                # Transform the bounding box coordinates
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
                                                
                                                # Flip horizontally as requested
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
                            if len(detections) == 0:
                                logger.debug("No detections found in any class")
                    elif hasattr(output_data, 'shape') and len(output_data.shape) == 3:
                        if output_data.shape[2] == 84:  # YOLOv8 format
                            # Process detections
                            for i in range(output_data.shape[1]):
                                bbox_data = output_data[0, i, :]
                                
                                # First 4 values are bbox (x_center, y_center, width, height)
                                x_center, y_center, w, h = bbox_data[:4]
                                
                                # Next 80 values are class scores
                                class_scores = bbox_data[4:]
                                max_score_idx = np.argmax(class_scores)
                                max_score = class_scores[max_score_idx]
                                
                                # Apply confidence threshold
                                if max_score > 0.25:  # Lower threshold for testing
                                    # Convert to normalized coordinates
                                    x1 = max(0, (x_center - w/2) / 640.0)
                                    y1 = max(0, (y_center - h/2) / 640.0)
                                    x2 = min(1, (x_center + w/2) / 640.0)
                                    y2 = min(1, (y_center + h/2) / 640.0)
                                    
                                    detections.append({
                                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                        'confidence': float(max_score),
                                        'class_id': int(max_score_idx),
                                        'label': COCO_CLASSES[max_score_idx] if max_score_idx < len(COCO_CLASSES) else 'unknown'
                                    })
                    break  # Process only first output
                    
            elif isinstance(raw_outputs, list):
                # YOLOv8 typically has one output tensor with shape [1, 8400, 84]
                # where 8400 is the number of anchor boxes and 84 = 4 (bbox) + 80 (classes)
                if len(raw_outputs) > 0:
                    output_data = raw_outputs[0]
                    logger.info(f"Raw output type: {type(output_data)}")
                    if hasattr(output_data, 'shape'):
                        logger.info(f"Output shape: {output_data.shape}")
                    else:
                        logger.info(f"Output is list with length: {len(output_data) if isinstance(output_data, list) else 'N/A'}")
                        if isinstance(output_data, list) and len(output_data) > 0:
                            logger.info(f"First element type: {type(output_data[0])}")
                            if hasattr(output_data[0], 'shape'):
                                logger.info(f"First element shape: {output_data[0].shape}")
                    
                    # Process YOLOv8 output format
                    if len(output_data.shape) == 3 and output_data.shape[2] == 84:
                        # Extract detections
                        for i in range(output_data.shape[1]):
                            bbox_data = output_data[0, i, :]
                            
                            # First 4 values are bbox coordinates (x_center, y_center, width, height)
                            x_center, y_center, w, h = bbox_data[:4]
                            
                            # Next 80 values are class scores
                            class_scores = bbox_data[4:]
                            max_score_idx = np.argmax(class_scores)
                            max_score = class_scores[max_score_idx]
                            
                            # Apply confidence threshold
                            if max_score > 0.5:
                                # Convert center format to corner format
                                x1 = (x_center - w/2) / 640.0  # Normalize to [0,1]
                                y1 = (y_center - h/2) / 640.0
                                x2 = (x_center + w/2) / 640.0
                                y2 = (y_center + h/2) / 640.0
                                
                                detections.append({
                                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                    'confidence': float(max_score),
                                    'class_id': int(max_score_idx),
                                    'label': COCO_CLASSES[max_score_idx] if max_score_idx < len(COCO_CLASSES) else 'unknown'
                                })
                    else:
                        logger.debug(f"Unexpected output shape: {output_data.shape}")
            else:
                # Handle dict format
                for output_name, output_data in raw_outputs.items():
                    logger.debug(f"Output {output_name} type: {type(output_data)}")
                    if hasattr(output_data, 'shape'):
                        logger.debug(f"Output {output_name} shape: {output_data.shape}")
                                    
        except Exception as e:
            logger.error(f"Output processing error: {e}")
            
        return detections[:10]  # Limit to top 10 detections
        
    def draw_detections(self, frame, detections):
        """Draw detection boxes and labels on frame"""
        annotated = frame.copy()
        h, w = frame.shape[:2]
        
        logger.debug(f"Frame dimensions for drawing: {w}x{h}")
        
        for det in detections:
            # bbox is already in [x1, y1, x2, y2] normalized format
            x1, y1, x2, y2 = det['bbox']
            
            # Log original normalized coordinates
            logger.debug(f"Normalized bbox: [{x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f}] for {det['label']}")
            
            # Convert to pixel coordinates
            x1 = int(x1 * w)
            y1 = int(y1 * h)
            x2 = int(x2 * w)
            y2 = int(y2 * h)
            
            # Log pixel coordinates
            logger.debug(f"Pixel bbox: [{x1}, {y1}, {x2}, {y2}] for {det['label']}")
            
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
            
            # In debug mode, also show bbox coordinates
            if stats.get('debug_mode', False):
                coord_text = f"[{x1},{y1},{x2},{y2}]"
                cv2.putText(annotated, coord_text, (x1, y2+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
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
        cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        cv2.putText(frame, "Hailo-8L YOLOv8s Detection", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {stats['fps']:.1f}", (20, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.putText(frame, f"Objects: {stats['objects']}", (150, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        cv2.putText(frame, f"Status: {stats['hailo_status']}", (20, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Frame: {frame.shape[1]}x{frame.shape[0]}", (20, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 255), 1)
        
        # Add debug markers to verify orientation
        if stats.get('debug_mode', False):
            h, w = frame.shape[:2]
            # Draw corner markers
            cv2.putText(frame, "TL", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "TR", (w-40, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "BL", (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "BR", (w-40, h-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # Draw center cross
            cv2.line(frame, (w//2-20, h//2), (w//2+20, h//2), (0, 255, 0), 2)
            cv2.line(frame, (w//2, h//2-20), (w//2, h//2+20), (0, 255, 0), 2)
                   
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
    <title>Hailo YOLO Detection</title>
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
    </style>
</head>
<body>
    <div class="container">
        <h1>Hailo-8L YOLOv8 Object Detection</h1>
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
                <div class="stat-value">YOLOv8s</div>
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

@app.route('/api/debug/toggle')
def toggle_debug():
    stats['debug_mode'] = not stats.get('debug_mode', False)
    return {'debug_mode': stats['debug_mode']}

# Global detector
detector = None

if __name__ == '__main__':
    logger.info("Starting Hailo Proper Web Detection...")
    
    # Check if running in Hailo environment
    if not os.path.exists('/home/pi/hailo-rpi5-examples'):
        logger.error("Hailo examples not found. Please install hailo-rpi5-examples")
        sys.exit(1)
        
    detector = HailoYOLODetector()
    detector.start()
    
    try:
        app.run(host='0.0.0.0', port=8080, debug=False)
    finally:
        if detector:
            detector.stop()