import asyncio
import logging
from typing import Optional
import numpy as np
import threading
import queue

logger = logging.getLogger(__name__)


class VideoPipeline:
    """Handles video capture and processing pipeline"""
    
    def __init__(self, config: dict):
        self.config = config
        self._camera = None
        self._capture_thread = None
        self._frame_queue = queue.Queue(maxsize=10)
        self._running = False
        
    async def initialize(self):
        """Initialize video pipeline"""
        camera_config = self.config['camera']
        
        try:
            # Import camera library based on platform
            try:
                from picamera2 import Picamera2
                self._use_picamera = True
                logger.info("Using PiCamera2")
            except ImportError:
                import cv2
                self._use_picamera = False
                logger.info("Using OpenCV camera")
                
            if self._use_picamera:
                # Initialize PiCamera2
                self._camera = Picamera2()
                config = self._camera.create_preview_configuration(
                    main={"size": tuple(camera_config['resolution']),
                          "format": "RGB888"},
                    controls={"FrameRate": camera_config['fps']}
                )
                self._camera.configure(config)
                self._camera.start()
            else:
                # Initialize OpenCV camera
                self._camera = cv2.VideoCapture(camera_config['device'])
                self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config['resolution'][0])
                self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config['resolution'][1])
                self._camera.set(cv2.CAP_PROP_FPS, camera_config['fps'])
                
            # Start capture thread
            self._running = True
            self._capture_thread = threading.Thread(target=self._capture_loop)
            self._capture_thread.start()
            
            logger.info("Video pipeline initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize camera: {e}")
            raise
            
    def _capture_loop(self):
        """Capture frames in separate thread"""
        while self._running:
            try:
                if self._use_picamera:
                    frame = self._camera.capture_array()
                    # Convert RGB to BGR for OpenCV compatibility
                    frame = frame[:, :, ::-1]
                else:
                    ret, frame = self._camera.read()
                    if not ret:
                        logger.warning("Failed to capture frame")
                        continue
                        
                # Add frame to queue (drop old frames if full)
                try:
                    self._frame_queue.put_nowait(frame)
                except queue.Full:
                    # Drop oldest frame
                    try:
                        self._frame_queue.get_nowait()
                        self._frame_queue.put_nowait(frame)
                    except queue.Empty:
                        pass
                        
            except Exception as e:
                logger.error(f"Error in capture loop: {e}")
                
    async def get_frame(self) -> Optional[np.ndarray]:
        """Get latest frame from pipeline"""
        try:
            frame = self._frame_queue.get_nowait()
            return frame
        except queue.Empty:
            return None
            
    async def stop(self):
        """Stop video pipeline"""
        logger.info("Stopping video pipeline...")
        self._running = False
        
        if self._capture_thread:
            self._capture_thread.join(timeout=2.0)
            
        if self._camera:
            if self._use_picamera:
                self._camera.stop()
            else:
                self._camera.release()
                
        logger.info("Video pipeline stopped")