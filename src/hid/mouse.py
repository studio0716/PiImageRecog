import struct
import time
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MouseHID:
    """USB HID Mouse device interface"""
    
    BUTTON_LEFT = 0x01
    BUTTON_RIGHT = 0x02
    BUTTON_MIDDLE = 0x04
    
    def __init__(self, device_path: str = "/dev/hidg0"):
        self.device_path = device_path
        self._device = None
        self._last_report = bytes(5)  # Store last report for relative movements
        
    def __enter__(self):
        self.open()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        
    def open(self):
        """Open the HID device"""
        try:
            self._device = open(self.device_path, 'wb', buffering=0)
            logger.info(f"Opened mouse HID device: {self.device_path}")
        except IOError as e:
            logger.error(f"Failed to open HID device {self.device_path}: {e}")
            raise
            
    def close(self):
        """Close the HID device"""
        if self._device:
            self._device.close()
            self._device = None
            
    def _send_report(self, report: bytes):
        """Send a 5-byte mouse report to the HID device"""
        if not self._device:
            raise RuntimeError("HID device not open")
            
        if len(report) != 5:
            raise ValueError(f"Mouse report must be 5 bytes, got {len(report)}")
            
        try:
            self._device.write(report)
            self._device.flush()
            self._last_report = report
        except IOError as e:
            logger.error(f"Failed to send HID report: {e}")
            raise
            
    def move(self, x: int, y: int, relative: bool = True):
        """
        Move the mouse cursor
        
        Args:
            x: X movement (-127 to 127 for relative, 0-65535 for absolute)
            y: Y movement (-127 to 127 for relative, 0-65535 for absolute)
            relative: If True, move relative to current position
        """
        if relative:
            # Clamp values to signed byte range
            x = max(-127, min(127, x))
            y = max(-127, min(127, y))
            
            # Report format: buttons, x, y, wheel, reserved
            report = struct.pack('bbbbb', 
                                self._last_report[0],  # Keep button state
                                x, y, 0, 0)
        else:
            # For absolute positioning, we'd need a different HID descriptor
            # This is a placeholder for future implementation
            raise NotImplementedError("Absolute positioning not yet implemented")
            
        self._send_report(report)
        
    def click(self, button: int = BUTTON_LEFT, duration: float = 0.1):
        """
        Perform a mouse click
        
        Args:
            button: Button to click (BUTTON_LEFT, BUTTON_RIGHT, BUTTON_MIDDLE)
            duration: How long to hold the button (seconds)
        """
        # Press button
        report = struct.pack('bbbbb', button, 0, 0, 0, 0)
        self._send_report(report)
        
        # Hold for duration
        time.sleep(duration)
        
        # Release button
        report = struct.pack('bbbbb', 0, 0, 0, 0, 0)
        self._send_report(report)
        
    def press(self, button: int = BUTTON_LEFT):
        """Press and hold a mouse button"""
        report = struct.pack('bbbbb', button, 0, 0, 0, 0)
        self._send_report(report)
        
    def release(self, button: int = BUTTON_LEFT):
        """Release a mouse button"""
        current_buttons = self._last_report[0] & ~button
        report = struct.pack('bbbbb', current_buttons, 0, 0, 0, 0)
        self._send_report(report)
        
    def scroll(self, amount: int):
        """
        Scroll the mouse wheel
        
        Args:
            amount: Scroll amount (-127 to 127, negative scrolls down)
        """
        amount = max(-127, min(127, amount))
        report = struct.pack('bbbbb', 
                            self._last_report[0],  # Keep button state
                            0, 0, amount, 0)
        self._send_report(report)
        
    def move_to(self, x: int, y: int, screen_width: int, screen_height: int,
                current_x: int, current_y: int, steps: int = 10):
        """
        Move mouse to absolute position using relative movements
        
        This simulates absolute positioning by calculating the required
        relative movements from the current position.
        
        Args:
            x, y: Target coordinates
            screen_width, screen_height: Screen dimensions
            current_x, current_y: Current mouse position
            steps: Number of steps to reach target (for smoothness)
        """
        # Calculate total movement needed
        dx = x - current_x
        dy = y - current_y
        
        # Move in steps for smooth motion
        for i in range(steps):
            # Calculate movement for this step
            step_x = dx // (steps - i)
            step_y = dy // (steps - i)
            
            # Update remaining movement
            dx -= step_x
            dy -= step_y
            
            # Send movement command
            self.move(step_x, step_y)
            time.sleep(0.01)  # Small delay for smoothness
            
    def drag(self, start_x: int, start_y: int, end_x: int, end_y: int,
             button: int = BUTTON_LEFT, steps: int = 20):
        """
        Perform a drag operation
        
        Args:
            start_x, start_y: Starting coordinates
            end_x, end_y: Ending coordinates
            button: Mouse button to hold during drag
            steps: Number of steps for smooth dragging
        """
        # Press button at start
        self.press(button)
        time.sleep(0.1)
        
        # Calculate movement
        dx = (end_x - start_x) / steps
        dy = (end_y - start_y) / steps
        
        # Perform drag
        for i in range(steps):
            self.move(int(dx), int(dy))
            time.sleep(0.02)
            
        # Release button
        self.release(button)
        
    def double_click(self, button: int = BUTTON_LEFT):
        """Perform a double click"""
        self.click(button, duration=0.05)
        time.sleep(0.05)
        self.click(button, duration=0.05)