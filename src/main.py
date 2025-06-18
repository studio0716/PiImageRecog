#!/usr/bin/env python3

import sys
import time
import logging
import argparse
import signal
from pathlib import Path
import yaml
import asyncio
from typing import Optional

# Add src to path
sys.path.append(str(Path(__file__).parent))

from hid.mouse import MouseHID
from hid.keyboard import KeyboardHID
from vision.screen_detector import ScreenDetector
from calibration.calibrator import Calibrator
from tracking.cursor_tracker import CursorTracker
from video_pipeline import VideoPipeline

# Setup logging
def setup_logging(config: dict):
    """Setup logging configuration"""
    log_config = config.get('logging', {})
    level = getattr(logging, log_config.get('level', 'INFO'))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_config.get('file', 'kcvm.log'))
        ]
    )
    
logger = logging.getLogger(__name__)


class KCVM:
    """Main KCVM application"""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.running = False
        
        # Initialize components
        self.mouse = None
        self.keyboard = None
        self.screen_detector = None
        self.cursor_tracker = None
        self.video_pipeline = None
        self.calibrator = None
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
            
    async def initialize(self):
        """Initialize all components"""
        logger.info("Initializing KCVM...")
        
        # Initialize HID devices
        try:
            self.mouse = MouseHID(self.config['advanced']['hid']['mouse_device'])
            self.mouse.open()
            logger.info("Mouse HID initialized")
        except Exception as e:
            logger.error(f"Failed to initialize mouse: {e}")
            raise
            
        try:
            self.keyboard = KeyboardHID(self.config['advanced']['hid']['keyboard_device'])
            self.keyboard.open()
            logger.info("Keyboard HID initialized")
        except Exception as e:
            logger.error(f"Failed to initialize keyboard: {e}")
            raise
            
        # Initialize vision components
        self.screen_detector = ScreenDetector(
            min_area=self.config['detection']['screen_detection']['min_area']
        )
        
        # Initialize video pipeline
        self.video_pipeline = VideoPipeline(self.config)
        await self.video_pipeline.initialize()
        
        # Initialize calibrator
        self.calibrator = Calibrator(self.config)
        
        # Initialize cursor tracker (will be created after calibration)
        self.cursor_tracker = None
        
        logger.info("KCVM initialized successfully")
        
    async def calibrate(self):
        """Run calibration process"""
        logger.info("Starting calibration...")
        
        if self.config['calibration']['auto_calibrate']:
            # Auto calibration
            success = await self.calibrator.auto_calibrate(
                self.video_pipeline,
                self.screen_detector,
                self.mouse
            )
            
            if not success:
                logger.warning("Auto calibration failed, falling back to manual")
                success = await self.calibrator.manual_calibrate(
                    self.video_pipeline,
                    self.screen_detector
                )
                
        else:
            # Manual calibration
            success = await self.calibrator.manual_calibrate(
                self.video_pipeline,
                self.screen_detector
            )
            
        if success:
            logger.info("Calibration completed successfully")
            # Initialize cursor tracker with calibration data
            self.cursor_tracker = CursorTracker(
                self.calibrator.get_screen_region()
            )
        else:
            logger.error("Calibration failed")
            raise RuntimeError("Failed to calibrate system")
            
    async def run(self):
        """Main application loop"""
        self.running = True
        logger.info("KCVM running - press Ctrl+C to stop")
        
        frame_count = 0
        last_fps_time = time.time()
        fps = 0
        
        try:
            while self.running:
                # Get frame from video pipeline
                frame = await self.video_pipeline.get_frame()
                if frame is None:
                    await asyncio.sleep(0.001)
                    continue
                    
                # Detect screen
                screen = self.screen_detector.detect(frame)
                if screen is None:
                    logger.warning("Screen not detected")
                    await asyncio.sleep(0.1)
                    continue
                    
                # Track cursor
                if self.cursor_tracker:
                    cursor_pos = self.cursor_tracker.track(frame, screen)
                    
                    # Update mouse position if needed
                    # (This is where gesture/input processing would go)
                    
                # Calculate FPS
                frame_count += 1
                current_time = time.time()
                if current_time - last_fps_time >= 1.0:
                    fps = frame_count / (current_time - last_fps_time)
                    frame_count = 0
                    last_fps_time = current_time
                    logger.debug(f"FPS: {fps:.1f}")
                    
                # Debug display if enabled
                if self.config['advanced']['debug']['show_preview']:
                    import cv2
                    debug_frame = self.screen_detector.draw_detection(frame, screen)
                    cv2.imshow("KCVM Debug", debug_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.running = False
                        
                # Control loop delay
                await asyncio.sleep(0.001)
                
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
        finally:
            await self.shutdown()
            
    async def shutdown(self):
        """Cleanup and shutdown"""
        logger.info("Shutting down KCVM...")
        self.running = False
        
        # Close HID devices
        if self.mouse:
            self.mouse.close()
        if self.keyboard:
            self.keyboard.close()
            
        # Stop video pipeline
        if self.video_pipeline:
            await self.video_pipeline.stop()
            
        # Close debug windows
        if self.config['advanced']['debug']['show_preview']:
            import cv2
            cv2.destroyAllWindows()
            
        logger.info("KCVM shutdown complete")
        

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="KCVM - Keyboard Camera Video Mouse")
    parser.add_argument(
        '--config',
        default='config/default.yaml',
        help='Configuration file path'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    parser.add_argument(
        '--calibrate',
        action='store_true',
        help='Force calibration'
    )
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Configuration file not found: {config_path}")
        sys.exit(1)
        
    # Create application
    app = KCVM(str(config_path))
    
    # Override debug setting if specified
    if args.debug:
        app.config['advanced']['debug']['show_preview'] = True
        app.config['logging']['level'] = 'DEBUG'
        
    # Setup logging
    setup_logging(app.config)
    
    try:
        # Initialize
        await app.initialize()
        
        # Calibrate
        if args.calibrate or not app.calibrator.is_calibrated():
            await app.calibrate()
            
        # Run
        await app.run()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
        

if __name__ == "__main__":
    # Setup signal handlers
    def signal_handler(sig, frame):
        logger.info("Received signal, shutting down...")
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run application
    asyncio.run(main())