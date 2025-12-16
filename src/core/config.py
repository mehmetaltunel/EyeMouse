"""
Eye Mouse - Configuration Module
Settings and constants with persistence support
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class CameraConfig:
    """Camera settings"""
    index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30


@dataclass  
class TrackingConfig:
    """Eye tracking settings"""
    ear_threshold: float = 0.21
    ear_consecutive_frames: int = 2
    blink_cooldown: float = 0.35


@dataclass
class MouseConfig:
    """Mouse control settings"""
    sensitivity: float = 2.0
    smoothing_samples: int = 5
    dead_zone: float = 0.015
    acceleration_curve: float = 1.5  # 1.0 = linear, >1 = exponential


@dataclass
class CalibrationConfig:
    """Calibration settings"""
    points_count: int = 9
    point_duration: float = 2.0
    margin_percent: float = 0.1


@dataclass
class AppConfig:
    """Main application configuration"""
    camera: CameraConfig = field(default_factory=CameraConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    mouse: MouseConfig = field(default_factory=MouseConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    
    # Runtime
    debug_mode: bool = False
    show_landmarks: bool = True
    
    # Paths
    config_file: str = "settings.json"
    calibration_file: str = "calibration.json"
    
    def save(self, filepath: Optional[str] = None):
        """Save configuration to file"""
        path = filepath or self.config_file
        data = {
            'camera': asdict(self.camera),
            'tracking': asdict(self.tracking),
            'mouse': asdict(self.mouse),
            'calibration': asdict(self.calibration),
            'debug_mode': self.debug_mode,
            'show_landmarks': self.show_landmarks
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
            
    def load(self, filepath: Optional[str] = None) -> bool:
        """Load configuration from file"""
        path = filepath or self.config_file
        if not os.path.exists(path):
            return False
            
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                
            if 'camera' in data:
                self.camera = CameraConfig(**data['camera'])
            if 'tracking' in data:
                self.tracking = TrackingConfig(**data['tracking'])
            if 'mouse' in data:
                self.mouse = MouseConfig(**data['mouse'])
            if 'calibration' in data:
                self.calibration = CalibrationConfig(**data['calibration'])
            if 'debug_mode' in data:
                self.debug_mode = data['debug_mode']
            if 'show_landmarks' in data:
                self.show_landmarks = data['show_landmarks']
                
            return True
        except Exception:
            return False


# Global config instance
config = AppConfig()
