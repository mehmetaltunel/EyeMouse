"""
Eye Mouse - Mouse Controller Module
Cursor control with acceleration curve and smoothing
"""

import pyautogui
import numpy as np
import math
from typing import Tuple, Optional
from collections import deque

# Optimize pyautogui for speed
pyautogui.PAUSE = 0
pyautogui.MINIMUM_DURATION = 0
pyautogui.FAILSAFE = False


class MouseController:
    """
    Mouse controller with advanced smoothing and acceleration
    """
    
    def __init__(self, sensitivity: float = 2.0,
                 smoothing_samples: int = 5,
                 dead_zone: float = 0.015,
                 acceleration_curve: float = 1.5):
        # PyAutoGUI settings
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.0
        
        # Screen dimensions
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Settings
        self.sensitivity = sensitivity
        self.dead_zone = dead_zone
        self.acceleration_curve = acceleration_curve
        
        # Smoothing buffers
        self._x_buffer = deque(maxlen=smoothing_samples)
        self._y_buffer = deque(maxlen=smoothing_samples)
        
        # State
        self._last_x: Optional[float] = None
        self._last_y: Optional[float] = None
        self._enabled = False
        
    def set_enabled(self, enabled: bool):
        """Enable/disable mouse control"""
        self._enabled = enabled
        if enabled:
            self.reset()
            
    def is_enabled(self) -> bool:
        """Check if mouse control is enabled"""
        return self._enabled
        
    def update_settings(self, sensitivity: float, dead_zone: float,
                        acceleration_curve: float, smoothing_samples: int):
        """Update controller settings"""
        self.sensitivity = sensitivity
        self.dead_zone = dead_zone
        self.acceleration_curve = acceleration_curve
        
        # Resize buffers
        new_x = deque(self._x_buffer, maxlen=smoothing_samples)
        new_y = deque(self._y_buffer, maxlen=smoothing_samples)
        self._x_buffer = new_x
        self._y_buffer = new_y
        
    def move_to_gaze(self, gaze_x: float, gaze_y: float):
        """
        Move cursor based on gaze position
        
        Args:
            gaze_x: Normalized X (0=looking left, 1=looking right)
            gaze_y: Normalized Y (0=looking up, 1=looking down)
        """
        if not self._enabled:
            return
            
        # Apply acceleration curve for better control
        # Center is 0.5, we want more precision near center
        cx, cy = 0.5, 0.5
        
        # Distance from center
        dx = gaze_x - cx
        dy = gaze_y - cy
        
        # Apply curve (exponent > 1 = more precision at center)
        if self.acceleration_curve != 1.0:
            sign_x = 1 if dx >= 0 else -1
            sign_y = 1 if dy >= 0 else -1
            dx = sign_x * (abs(dx) ** self.acceleration_curve)
            dy = sign_y * (abs(dy) ** self.acceleration_curve)
        
        # Apply sensitivity
        dx *= self.sensitivity
        dy *= self.sensitivity
        
        # Map to screen coordinates
        target_x = (cx + dx) * self.screen_width
        target_y = (cy + dy) * self.screen_height
        
        # Clamp to screen
        target_x = max(0, min(self.screen_width - 1, target_x))
        target_y = max(0, min(self.screen_height - 1, target_y))
        
        # Add to smoothing buffer
        self._x_buffer.append(target_x)
        self._y_buffer.append(target_y)
        
        # Calculate smoothed position
        smooth_x = sum(self._x_buffer) / len(self._x_buffer)
        smooth_y = sum(self._y_buffer) / len(self._y_buffer)
        
        # Dead zone check
        if self._last_x is not None and self._last_y is not None:
            delta = math.sqrt(
                ((smooth_x - self._last_x) / self.screen_width) ** 2 +
                ((smooth_y - self._last_y) / self.screen_height) ** 2
            )
            if delta < self.dead_zone:
                return
        
        # Move cursor
        try:
            pyautogui.moveTo(int(smooth_x), int(smooth_y), _pause=False)
            self._last_x = smooth_x
            self._last_y = smooth_y
        except Exception:
            pass
            
    def click(self, button: str = 'left'):
        """Perform click"""
        if not self._enabled:
            return
        try:
            pyautogui.click(button=button, _pause=False)
        except Exception:
            pass
            
    def double_click(self):
        """Perform double click"""
        if not self._enabled:
            return
        try:
            pyautogui.doubleClick(_pause=False)
        except Exception:
            pass
            
    def reset(self):
        """Reset smoothing state"""
        self._x_buffer.clear()
        self._y_buffer.clear()
        self._last_x = None
        self._last_y = None
        
    def get_screen_size(self) -> Tuple[int, int]:
        """Get screen dimensions"""
        return (self.screen_width, self.screen_height)
        
    def get_position(self) -> Tuple[int, int]:
        """Get current cursor position"""
        return pyautogui.position()
