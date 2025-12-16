"""
Eye Mouse - Calibration Module
Multi-point calibration with gaze validation
"""

import cv2
import numpy as np
import json
import os
import time
import math
from typing import Optional, Tuple, List
from dataclasses import dataclass, asdict
from enum import Enum, auto


class GazeQuality(Enum):
    """Quality of current gaze relative to target"""
    NO_GAZE = auto()      # Face not detected
    OFF_TARGET = auto()   # Looking elsewhere
    NEAR_TARGET = auto()  # Close to target
    ON_TARGET = auto()    # Looking at target


@dataclass
class CalibrationPoint:
    """Single calibration point data"""
    screen_x: int
    screen_y: int
    gaze_x: float
    gaze_y: float


class Calibrator:
    """
    Improved calibration system with gaze validation
    Only accepts samples when user is actually looking at the target
    """
    
    def __init__(self, screen_width: int, screen_height: int,
                 points_count: int = 9,
                 hold_duration: float = 2.0,
                 margin_percent: float = 0.12):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.points_count = points_count
        self.hold_duration = hold_duration  # Time to hold gaze ON target
        self.margin_percent = margin_percent
        
        # Tolerance for "on target" (as fraction of screen)
        # Higher value = more lenient (easier to hit target)
        self.initial_tolerance = 0.35  # 35% of screen diagonal
        
        # Requirements
        self.min_samples = 20
        
        # Generate grid
        self._target_points = self._generate_grid()
        
        # State
        self._collected_points: List[CalibrationPoint] = []
        self._gaze_samples: List[Tuple[float, float]] = []
        self._on_target_time = 0.0
        self._last_update_time = 0.0
        self._gaze_quality = GazeQuality.NO_GAZE
        
        self._is_active = False
        self._current_index = 0
        
        # Transform
        self._transform: Optional[np.ndarray] = None
        
    def _generate_grid(self) -> List[Tuple[int, int]]:
        """Generate calibration point grid"""
        points = []
        margin_x = int(self.screen_width * self.margin_percent)
        margin_y = int(self.screen_height * self.margin_percent)
        
        if self.points_count == 9:
            rows, cols = 3, 3
        elif self.points_count == 5:
            return [
                (self.screen_width // 2, self.screen_height // 2),
                (margin_x, margin_y),
                (self.screen_width - margin_x, margin_y),
                (margin_x, self.screen_height - margin_y),
                (self.screen_width - margin_x, self.screen_height - margin_y)
            ]
        else:
            rows = cols = int(np.sqrt(self.points_count))
            
        inner_w = self.screen_width - 2 * margin_x
        inner_h = self.screen_height - 2 * margin_y
        
        for row in range(rows):
            for col in range(cols):
                x = margin_x + int(inner_w * col / max(1, cols - 1)) if cols > 1 else self.screen_width // 2
                y = margin_y + int(inner_h * row / max(1, rows - 1)) if rows > 1 else self.screen_height // 2
                points.append((x, y))
                
        return points
    
    def start(self):
        """Start calibration process"""
        self._is_active = True
        self._current_index = 0
        self._collected_points = []
        self._gaze_samples = []
        self._on_target_time = 0.0
        self._last_update_time = time.time()
        self._gaze_quality = GazeQuality.NO_GAZE
        
    def stop(self):
        """Stop calibration"""
        self._is_active = False
        
    def is_active(self) -> bool:
        return self._is_active
        
    def get_gaze_quality(self) -> GazeQuality:
        return self._gaze_quality
    
    def _is_gaze_on_target(self, gaze: Tuple[float, float], 
                           target: Tuple[int, int]) -> Tuple[bool, float]:
        """
        Check if gaze is pointing at the target
        
        For initial calibration (no transform yet), we use a simple
        estimation based on the expected linear relationship.
        
        Returns: (is_on_target, distance_normalized)
        """
        # Normalize target to 0-1
        target_norm_x = target[0] / self.screen_width
        target_norm_y = target[1] / self.screen_height
        
        # Calculate distance between gaze and target (in normalized space)
        # Note: gaze is already 0-1 normalized
        dx = gaze[0] - target_norm_x
        dy = gaze[1] - target_norm_y
        distance = math.sqrt(dx * dx + dy * dy)
        
        # Use initial tolerance (lenient since not calibrated)
        tolerance = self.initial_tolerance
        
        return distance < tolerance, distance
        
    def update(self, gaze: Optional[Tuple[float, float]]) -> Tuple[bool, Optional[Tuple[int, int]], float]:
        """
        Update calibration with new gaze sample
        
        Returns: (completed, current_target_point, progress)
        """
        if not self._is_active:
            return True, None, 1.0
            
        if self._current_index >= len(self._target_points):
            self._finalize()
            return True, None, 1.0
            
        current_target = self._target_points[self._current_index]
        current_time = time.time()
        dt = current_time - self._last_update_time
        self._last_update_time = current_time
        
        # Check gaze quality
        if gaze is None:
            self._gaze_quality = GazeQuality.NO_GAZE
            # Don't reset progress, just pause
        else:
            is_on_target, distance = self._is_gaze_on_target(gaze, current_target)
            
            if is_on_target:
                self._gaze_quality = GazeQuality.ON_TARGET
                self._on_target_time += dt
                self._gaze_samples.append(gaze)
            elif distance < self.initial_tolerance * 1.5:
                self._gaze_quality = GazeQuality.NEAR_TARGET
                # Close but not accepting samples
            else:
                self._gaze_quality = GazeQuality.OFF_TARGET
                # Way off - could reset, but let's be lenient
        
        progress = min(1.0, self._on_target_time / self.hold_duration)
        
        # Point completed
        if self._on_target_time >= self.hold_duration and len(self._gaze_samples) >= self.min_samples:
            # Calculate average gaze
            avg_x = sum(g[0] for g in self._gaze_samples) / len(self._gaze_samples)
            avg_y = sum(g[1] for g in self._gaze_samples) / len(self._gaze_samples)
            
            self._collected_points.append(CalibrationPoint(
                screen_x=current_target[0],
                screen_y=current_target[1],
                gaze_x=avg_x,
                gaze_y=avg_y
            ))
            
            # Next point
            self._current_index += 1
            self._gaze_samples = []
            self._on_target_time = 0.0
            self._gaze_quality = GazeQuality.NO_GAZE
            
            if self._current_index >= len(self._target_points):
                self._finalize()
                return True, None, 1.0
                
        return False, current_target, progress
    
    def _finalize(self):
        """Finalize calibration and compute transform"""
        self._is_active = False
        
        if len(self._collected_points) < 4:
            return
            
        src = np.array([[p.gaze_x, p.gaze_y] for p in self._collected_points], dtype=np.float32)
        dst = np.array([[p.screen_x, p.screen_y] for p in self._collected_points], dtype=np.float32)
        
        self._transform, _ = cv2.findHomography(src, dst)
        
    def transform_gaze(self, gaze: Tuple[float, float]) -> Tuple[int, int]:
        """Transform gaze to screen coordinates"""
        if self._transform is None:
            x = int(gaze[0] * self.screen_width)
            y = int(gaze[1] * self.screen_height)
        else:
            pt = np.array([[[gaze[0], gaze[1]]]], dtype=np.float32)
            result = cv2.perspectiveTransform(pt, self._transform)
            x = int(np.clip(result[0, 0, 0], 0, self.screen_width - 1))
            y = int(np.clip(result[0, 0, 1], 0, self.screen_height - 1))
            
        return (x, y)
    
    def is_calibrated(self) -> bool:
        return self._transform is not None
        
    def save(self, filepath: str) -> bool:
        if self._transform is None:
            return False
            
        data = {
            'transform': self._transform.tolist(),
            'points': [asdict(p) for p in self._collected_points],
            'screen_size': [self.screen_width, self.screen_height]
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception:
            return False
            
    def load(self, filepath: str) -> bool:
        if not os.path.exists(filepath):
            return False
            
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            self._transform = np.array(data['transform'], dtype=np.float32)
            self._collected_points = [
                CalibrationPoint(**p) for p in data.get('points', [])
            ]
            return True
        except Exception:
            return False
            
    def get_progress_info(self) -> Tuple[int, int]:
        return (self._current_index + 1, len(self._target_points))
