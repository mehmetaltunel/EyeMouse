"""
Eye Mouse - Blink Detector Module
EAR (Eye Aspect Ratio) based blink detection
"""

import time
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum, auto
from collections import deque


class BlinkType(Enum):
    """Blink types for click mapping"""
    NONE = auto()
    LEFT = auto()
    RIGHT = auto()
    BOTH = auto()


@dataclass
class BlinkEvent:
    """Blink event data"""
    blink_type: BlinkType
    timestamp: float
    left_ear: float
    right_ear: float


class BlinkDetector:
    """
    Eye blink detector using Eye Aspect Ratio (EAR)
    
    Detects single-eye blinks for left/right click distinction
    """
    
    def __init__(self, ear_threshold: float = 0.21,
                 consecutive_frames: int = 2,
                 cooldown: float = 0.35,
                 smoothing_samples: int = 3):
        self.ear_threshold = ear_threshold
        self.consecutive_frames = consecutive_frames
        self.cooldown = cooldown
        
        # State tracking
        self._left_closed_count = 0
        self._right_closed_count = 0
        self._left_was_closed = False
        self._right_was_closed = False
        self._last_blink_time = 0.0
        
        # EAR smoothing
        self._left_ear_buffer = deque(maxlen=smoothing_samples)
        self._right_ear_buffer = deque(maxlen=smoothing_samples)
        
        # Last values
        self._left_ear = 0.0
        self._right_ear = 0.0
        
    def update_thresholds(self, ear_threshold: float, cooldown: float):
        """Update detection thresholds"""
        self.ear_threshold = ear_threshold
        self.cooldown = cooldown
        
    def detect(self, left_landmarks: List, right_landmarks: List) -> Optional[BlinkEvent]:
        """
        Detect blink from eye landmarks
        
        Args:
            left_landmarks: Left eye landmarks (6 points)
            right_landmarks: Right eye landmarks (6 points)
            
        Returns:
            BlinkEvent if blink detected, None otherwise
        """
        now = time.time()
        
        # Cooldown check
        if now - self._last_blink_time < self.cooldown:
            return None
            
        # Calculate EAR
        left_ear_raw = self._calculate_ear(left_landmarks)
        right_ear_raw = self._calculate_ear(right_landmarks)
        
        # Smooth
        self._left_ear_buffer.append(left_ear_raw)
        self._right_ear_buffer.append(right_ear_raw)
        
        self._left_ear = sum(self._left_ear_buffer) / len(self._left_ear_buffer)
        self._right_ear = sum(self._right_ear_buffer) / len(self._right_ear_buffer)
        
        # Detect closed state
        left_closed = self._left_ear < self.ear_threshold
        right_closed = self._right_ear < self.ear_threshold
        
        # Single eye blink detection
        blink_type = self._check_blinks(left_closed, right_closed)
        
        if blink_type != BlinkType.NONE:
            self._last_blink_time = now
            return BlinkEvent(
                blink_type=blink_type,
                timestamp=now,
                left_ear=self._left_ear,
                right_ear=self._right_ear
            )
            
        return None
    
    def _calculate_ear(self, landmarks: List) -> float:
        """
        Calculate Eye Aspect Ratio
        
        EAR = (vertical1 + vertical2) / (2 * horizontal)
        """
        if len(landmarks) < 6:
            return 0.5
            
        pts = np.array([(p[0], p[1]) for p in landmarks])
        
        # Vertical distances
        v1 = np.linalg.norm(pts[1] - pts[5])
        v2 = np.linalg.norm(pts[2] - pts[4])
        
        # Horizontal distance
        h = np.linalg.norm(pts[0] - pts[3])
        
        if h < 1:
            return 0.5
            
        return (v1 + v2) / (2.0 * h)
    
    def _check_blinks(self, left_closed: bool, right_closed: bool) -> BlinkType:
        """Check for blink events"""
        result = BlinkType.NONE
        
        # Left eye blink (right eye must be open)
        if left_closed and not right_closed:
            self._left_closed_count += 1
            self._left_was_closed = True
        elif self._left_was_closed and not left_closed:
            if self._left_closed_count >= self.consecutive_frames:
                result = BlinkType.LEFT
            self._left_closed_count = 0
            self._left_was_closed = False
        
        # Right eye blink (left eye must be open)
        if right_closed and not left_closed:
            self._right_closed_count += 1
            self._right_was_closed = True
        elif self._right_was_closed and not right_closed:
            if self._right_closed_count >= self.consecutive_frames:
                if result == BlinkType.LEFT:
                    result = BlinkType.BOTH
                else:
                    result = BlinkType.RIGHT
            self._right_closed_count = 0
            self._right_was_closed = False
            
        # Both eyes closed
        if left_closed and right_closed:
            self._left_closed_count += 1
            self._right_closed_count += 1
        elif self._left_was_closed and self._right_was_closed:
            if not left_closed and not right_closed:
                min_count = min(self._left_closed_count, self._right_closed_count)
                if min_count >= self.consecutive_frames:
                    result = BlinkType.BOTH
                self._reset()
                    
        return result
    
    def _reset(self):
        """Reset state"""
        self._left_closed_count = 0
        self._right_closed_count = 0
        self._left_was_closed = False
        self._right_was_closed = False
        
    def get_ear_values(self) -> Tuple[float, float]:
        """Get current EAR values"""
        return (self._left_ear, self._right_ear)
    
    def reset(self):
        """Full reset"""
        self._reset()
        self._left_ear_buffer.clear()
        self._right_ear_buffer.clear()
        self._last_blink_time = 0.0
