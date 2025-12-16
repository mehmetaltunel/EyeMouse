"""
Eye Mouse - Nose Position Tracker
Simple and reliable: tracks nose tip position in camera frame
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass 
class GazeData:
    """Tracking data - compatible with rest of app"""
    position: Tuple[float, float]  # 0-1 normalized screen position
    nose_x: float = 0.0  # Raw nose X (0-1)
    nose_y: float = 0.0  # Raw nose Y (0-1)
    distance: float = 0.0  # Estimated distance (normalized, 1.0 ~ 50cm)
    confidence: float = 0.0


class EyeTracker:
    """
    Simple Nose Position Tracker
    
    How it works:
    - Tracks nose tip position in camera frame
    - Maps nose position to screen coordinates
    - Uses calibration to adjust mapping
    
    Much more reliable than head rotation!
    """
    
    # Key landmarks
    NOSE_TIP = 1
    
    # Calibration bounds (normalized 0-1)
    # These define the "active area" of the camera frame that maps to the screen
    # Smaller area = Higher sensitivity (less head movement needed)
    _base_min_x: float = 0.3
    _base_max_x: float = 0.7
    _base_min_y: float = 0.3
    _base_max_y: float = 0.7
    
    _min_x: float = 0.3
    _max_x: float = 0.7
    _min_y: float = 0.3
    _max_y: float = 0.7
    
    _sensitivity: float = 1.0
    
    # Eye contour for blink detection
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]
    
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self._param_history = []
        
        self._landmarks: List = []
        self._frame_size: Tuple[int, int] = (640, 480)
        self._gaze_data: Optional[GazeData] = None
        
        # Calibration bounds (will be set during calibration)
        # These define the "active area" where nose movement maps to screen
        self._min_x = 0.3  # Nose at 30% from left = screen left edge
        self._max_x = 0.7  # Nose at 70% from left = screen right edge
        self._min_y = 0.3
        self._max_y = 0.7
        
        # Smoothing
        self._smooth_x = 0.5
        self._smooth_y = 0.5
        self._smooth_factor = 0.12  # Lower = smoother
        
        # History for extra smoothing
        self._history_x = []
        self._history_y = []
        self._history_size = 8

    def set_sensitivity(self, value: float):
        """
        Adjust sensitivity by shrinking/expanding the active area.
        value: 1.0 (normal) to 10.0 (high sensitivity)
        """
        self._sensitivity = max(1.0, min(10.0, value))
        
        # Calculate new range width/height based on sensitivity
        # Higher sensitivity = Smaller range
        base_w = self._base_max_x - self._base_min_x
        base_h = self._base_max_y - self._base_min_y
        
        new_w = base_w / self._sensitivity
        new_h = base_h / self._sensitivity
        
        # Center point
        cx = (self._base_min_x + self._base_max_x) / 2
        cy = (self._base_min_y + self._base_max_y) / 2
        
        # Update bounds
        self._min_x = cx - new_w / 2
        self._max_x = cx + new_w / 2
        self._min_y = cy - new_h / 2
        self._max_y = cy + new_h / 2
        
    def process(self, frame: np.ndarray) -> Tuple[bool, Optional[GazeData]]:
        """Process frame and track nose position"""
        h, w = frame.shape[:2]
        self._frame_size = (w, h)
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        
        if not results.multi_face_landmarks:
            self._landmarks = []
            self._gaze_data = None
            return False, None
            
        face = results.multi_face_landmarks[0]
        self._landmarks = [(lm.x, lm.y, lm.z) for lm in face.landmark]
        
        self._gaze_data = self._calculate_position()
        return True, self._gaze_data
    
    def _calculate_distance(self) -> float:
        """Estimate relative distance based on eye width"""
        if not self._landmarks:
            return 0.0
            
        # Use outer eye corners
        left_outer = self._landmarks[33]   # Left eye outer corner
        right_outer = self._landmarks[263] # Right eye outer corner
        
        # Calculate distance in normalized coordinates
        dx = left_outer[0] - right_outer[0]
        dy = left_outer[1] - right_outer[1]
        dz = left_outer[2] - right_outer[2]
        
        # Euclidean distance
        dist_pixels = (dx * dx + dy * dy + dz * dz) ** 0.5
        
        # Heuristic: approx 0.15 normalized width is "normal" distance (~50cm)
        # Larger width = closer, Smaller = further
        # Invert so larger value = further away
        if dist_pixels < 0.01:
            return 0.0
            
        return 0.15 / dist_pixels
    
    def _calculate_position(self) -> GazeData:
        """Calculate screen position from nose tip"""
        # Get nose position (already 0-1 normalized by MediaPipe)
        nose = self._landmarks[self.NOSE_TIP]
        raw_x = nose[0]  # 0 = left of frame, 1 = right
        raw_y = nose[1]  # 0 = top, 1 = bottom
        
        # Add to history for smoothing
        self._history_x.append(raw_x)
        self._history_y.append(raw_y)
        if len(self._history_x) > self._history_size:
            self._history_x.pop(0)
            self._history_y.pop(0)
        
        # Average smoothing
        avg_x = sum(self._history_x) / len(self._history_x)
        avg_y = sum(self._history_y) / len(self._history_y)
        
        # Map nose position to screen (0-1)
        # Camera is mirrored (flipped horizontally)
        # Physical Left -> Left in frame (decreasing X) -> Cursor Left (decreasing X)
        # So NO inversion needed for mirrored camera
        screen_x = self._map_range(avg_x, self._min_x, self._max_x)
        screen_y = self._map_range(avg_y, self._min_y, self._max_y)
        
        # Clamp
        screen_x = max(0.0, min(1.0, screen_x))
        screen_y = max(0.0, min(1.0, screen_y))
        
        # Exponential smoothing
        self._smooth_x += (screen_x - self._smooth_x) * self._smooth_factor
        self._smooth_y += (screen_y - self._smooth_y) * self._smooth_factor
        
        # Calculate distance
        dist = self._calculate_distance()
        
        return GazeData(
            position=(self._smooth_x, self._smooth_y),
            nose_x=avg_x,
            nose_y=avg_y,
            distance=dist,
            confidence=1.0
        )
    
    def _map_range(self, value: float, min_val: float, max_val: float) -> float:
        """Map value from [min_val, max_val] to [0, 1]"""
        if max_val <= min_val:
            return 0.5
        return (value - min_val) / (max_val - min_val)
    
    def set_calibration_bounds(self, min_x: float, max_x: float, 
                                min_y: float, max_y: float):
        """Set the nose position bounds for screen mapping"""
        self._min_x = min_x
        self._max_x = max_x
        self._min_y = min_y
        self._max_y = max_y
        
    def get_calibration_bounds(self) -> Tuple[float, float, float, float]:
        """Get current calibration bounds"""
        return (self._min_x, self._max_x, self._min_y, self._max_y)
    
    def get_eye_landmarks(self, eye: str) -> List[Tuple[float, float, float]]:
        """Get eye landmarks for EAR (blink detection)"""
        indices = self.LEFT_EYE if eye == 'left' else self.RIGHT_EYE
        if not self._landmarks or len(self._landmarks) <= max(indices):
            return []
        # Convert to pixel coordinates
        w, h = self._frame_size
        return [(self._landmarks[i][0] * w, 
                 self._landmarks[i][1] * h, 
                 self._landmarks[i][2] * w) for i in indices]
    
    def draw_overlay(self, frame: np.ndarray, show_landmarks: bool = True) -> np.ndarray:
        """Draw tracking overlay"""
        if not self._gaze_data:
            return frame
            
        h, w = frame.shape[:2]
        
        # Draw nose tip
        if self._landmarks:
            nose = self._landmarks[self.NOSE_TIP]
            pt = (int(nose[0] * w), int(nose[1] * h))
            
            # Crosshair
            cv2.line(frame, (pt[0] - 15, pt[1]), (pt[0] + 15, pt[1]), (0, 255, 0), 2)
            cv2.line(frame, (pt[0], pt[1] - 15), (pt[0], pt[1] + 15), (0, 255, 0), 2)
            cv2.circle(frame, pt, 5, (0, 255, 0), -1)
        
        # Draw calibration bounds box
        if show_landmarks:
            x1 = int(self._min_x * w)
            x2 = int(self._max_x * w)
            y1 = int(self._min_y * h)
            y2 = int(self._max_y * h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 255), 1)
        
        # Info text
        gx, gy = self._gaze_data.position
        nx, ny = self._gaze_data.nose_x, self._gaze_data.nose_y
        text = f"Screen: ({gx:.2f}, {gy:.2f}) Nose: ({nx:.2f}, {ny:.2f})"
        cv2.putText(frame, text, (10, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Draw eye contours
        if show_landmarks:
            for indices in [self.LEFT_EYE, self.RIGHT_EYE]:
                pts = []
                for i in indices:
                    if i < len(self._landmarks):
                        px = int(self._landmarks[i][0] * w)
                        py = int(self._landmarks[i][1] * h)
                        pts.append((px, py))
                if len(pts) >= 4:
                    cv2.polylines(frame, [np.array(pts)], True, (200, 200, 0), 1)
        
        return frame
    
    def close(self):
        self.face_mesh.close()
