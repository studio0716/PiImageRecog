import struct
import time
from typing import List, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class KeyboardHID:
    """USB HID Keyboard device interface"""
    
    # Modifier keys
    MOD_NONE = 0x00
    MOD_LCTRL = 0x01
    MOD_LSHIFT = 0x02
    MOD_LALT = 0x04
    MOD_LMETA = 0x08  # Windows/Command key
    MOD_RCTRL = 0x10
    MOD_RSHIFT = 0x20
    MOD_RALT = 0x40
    MOD_RMETA = 0x80
    
    # Common key codes (USB HID usage table)
    KEY_CODES = {
        'a': 0x04, 'b': 0x05, 'c': 0x06, 'd': 0x07, 'e': 0x08,
        'f': 0x09, 'g': 0x0a, 'h': 0x0b, 'i': 0x0c, 'j': 0x0d,
        'k': 0x0e, 'l': 0x0f, 'm': 0x10, 'n': 0x11, 'o': 0x12,
        'p': 0x13, 'q': 0x14, 'r': 0x15, 's': 0x16, 't': 0x17,
        'u': 0x18, 'v': 0x19, 'w': 0x1a, 'x': 0x1b, 'y': 0x1c,
        'z': 0x1d,
        '1': 0x1e, '2': 0x1f, '3': 0x20, '4': 0x21, '5': 0x22,
        '6': 0x23, '7': 0x24, '8': 0x25, '9': 0x26, '0': 0x27,
        '\n': 0x28,  # Enter
        '\x1b': 0x29,  # Escape
        '\b': 0x2a,  # Backspace
        '\t': 0x2b,  # Tab
        ' ': 0x2c,  # Space
        '-': 0x2d, '=': 0x2e, '[': 0x2f, ']': 0x30, '\\': 0x31,
        ';': 0x33, "'": 0x34, '`': 0x35, ',': 0x36, '.': 0x37,
        '/': 0x38,
        # Function keys
        'F1': 0x3a, 'F2': 0x3b, 'F3': 0x3c, 'F4': 0x3d,
        'F5': 0x3e, 'F6': 0x3f, 'F7': 0x40, 'F8': 0x41,
        'F9': 0x42, 'F10': 0x43, 'F11': 0x44, 'F12': 0x45,
        # Special keys
        'PRINT': 0x46, 'SCROLL': 0x47, 'PAUSE': 0x48,
        'INSERT': 0x49, 'HOME': 0x4a, 'PGUP': 0x4b,
        'DELETE': 0x4c, 'END': 0x4d, 'PGDOWN': 0x4e,
        'RIGHT': 0x4f, 'LEFT': 0x50, 'DOWN': 0x51, 'UP': 0x52,
    }
    
    # Shifted characters mapping
    SHIFT_CHARS = {
        '!': '1', '@': '2', '#': '3', '$': '4', '%': '5',
        '^': '6', '&': '7', '*': '8', '(': '9', ')': '0',
        '_': '-', '+': '=', '{': '[', '}': ']', '|': '\\',
        ':': ';', '"': "'", '<': ',', '>': '.', '?': '/',
        '~': '`',
    }
    
    def __init__(self, device_path: str = "/dev/hidg1"):
        self.device_path = device_path
        self._device = None
        self._caps_lock = False
        
    def __enter__(self):
        self.open()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        
    def open(self):
        """Open the HID device"""
        try:
            self._device = open(self.device_path, 'wb', buffering=0)
            logger.info(f"Opened keyboard HID device: {self.device_path}")
        except IOError as e:
            logger.error(f"Failed to open HID device {self.device_path}: {e}")
            raise
            
    def close(self):
        """Close the HID device"""
        if self._device:
            self._device.close()
            self._device = None
            
    def _send_report(self, report: bytes):
        """Send an 8-byte keyboard report to the HID device"""
        if not self._device:
            raise RuntimeError("HID device not open")
            
        if len(report) != 8:
            raise ValueError(f"Keyboard report must be 8 bytes, got {len(report)}")
            
        try:
            self._device.write(report)
            self._device.flush()
        except IOError as e:
            logger.error(f"Failed to send HID report: {e}")
            raise
            
    def _char_to_keycode(self, char: str) -> tuple[int, int]:
        """
        Convert a character to its keycode and modifier
        
        Returns:
            (keycode, modifier)
        """
        # Check if it's a shifted character
        if char in self.SHIFT_CHARS:
            base_char = self.SHIFT_CHARS[char]
            return self.KEY_CODES.get(base_char, 0), self.MOD_LSHIFT
            
        # Check if it's an uppercase letter
        if char.isupper():
            return self.KEY_CODES.get(char.lower(), 0), self.MOD_LSHIFT
            
        # Normal character
        return self.KEY_CODES.get(char, 0), self.MOD_NONE
        
    def press_key(self, keycode: int, modifiers: int = MOD_NONE):
        """
        Press a key with optional modifiers
        
        Args:
            keycode: USB HID keycode
            modifiers: Modifier keys to hold
        """
        # Report format: modifier, reserved, key1-key6
        report = struct.pack('BBBBBBBB', modifiers, 0, keycode, 0, 0, 0, 0, 0)
        self._send_report(report)
        
    def release_all(self):
        """Release all keys"""
        report = bytes(8)
        self._send_report(report)
        
    def tap_key(self, keycode: int, modifiers: int = MOD_NONE, duration: float = 0.05):
        """Press and release a key"""
        self.press_key(keycode, modifiers)
        time.sleep(duration)
        self.release_all()
        
    def type_char(self, char: str):
        """Type a single character"""
        keycode, modifier = self._char_to_keycode(char)
        if keycode:
            self.tap_key(keycode, modifier)
        else:
            logger.warning(f"Unknown character: {char}")
            
    def type_string(self, text: str, delay: float = 0.01):
        """
        Type a string of text
        
        Args:
            text: Text to type
            delay: Delay between keystrokes (seconds)
        """
        for char in text:
            self.type_char(char)
            time.sleep(delay)
            
    def key_combination(self, modifiers: int, key: str):
        """
        Press a key combination (e.g., Ctrl+C)
        
        Args:
            modifiers: Modifier keys (can be OR'd together)
            key: Key to press with modifiers
        """
        keycode = self.KEY_CODES.get(key.lower(), 0)
        if keycode:
            self.tap_key(keycode, modifiers)
        else:
            logger.warning(f"Unknown key: {key}")
            
    def press_multiple(self, keys: List[str], modifiers: int = MOD_NONE):
        """
        Press multiple keys simultaneously (up to 6)
        
        Args:
            keys: List of keys to press
            modifiers: Modifier keys to hold
        """
        if len(keys) > 6:
            logger.warning("Can only press up to 6 keys simultaneously")
            keys = keys[:6]
            
        # Get keycodes
        keycodes = []
        for key in keys:
            if key.lower() in self.KEY_CODES:
                keycodes.append(self.KEY_CODES[key.lower()])
            else:
                logger.warning(f"Unknown key: {key}")
                
        # Pad with zeros
        keycodes.extend([0] * (6 - len(keycodes)))
        
        # Send report
        report = struct.pack('BBBBBBBB', modifiers, 0, *keycodes)
        self._send_report(report)
        
    def special_key(self, key_name: str):
        """
        Press a special key by name
        
        Args:
            key_name: Name of special key (e.g., 'HOME', 'END', 'F1')
        """
        if key_name in self.KEY_CODES:
            self.tap_key(self.KEY_CODES[key_name])
        else:
            logger.warning(f"Unknown special key: {key_name}")
            
    # Convenience methods for common operations
    
    def copy(self):
        """Ctrl+C"""
        self.key_combination(self.MOD_LCTRL, 'c')
        
    def paste(self):
        """Ctrl+V"""
        self.key_combination(self.MOD_LCTRL, 'v')
        
    def cut(self):
        """Ctrl+X"""
        self.key_combination(self.MOD_LCTRL, 'x')
        
    def select_all(self):
        """Ctrl+A"""
        self.key_combination(self.MOD_LCTRL, 'a')
        
    def undo(self):
        """Ctrl+Z"""
        self.key_combination(self.MOD_LCTRL, 'z')
        
    def redo(self):
        """Ctrl+Y or Ctrl+Shift+Z depending on platform"""
        self.key_combination(self.MOD_LCTRL, 'y')
        
    def save(self):
        """Ctrl+S"""
        self.key_combination(self.MOD_LCTRL, 's')
        
    def find(self):
        """Ctrl+F"""
        self.key_combination(self.MOD_LCTRL, 'f')
        
    def tab_switch(self, forward: bool = True):
        """Switch tabs (Ctrl+Tab or Ctrl+Shift+Tab)"""
        if forward:
            self.key_combination(self.MOD_LCTRL, '\t')
        else:
            self.key_combination(self.MOD_LCTRL | self.MOD_LSHIFT, '\t')
            
    def alt_tab(self):
        """Alt+Tab for window switching"""
        self.key_combination(self.MOD_LALT, '\t')