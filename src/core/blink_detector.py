"""
Göz Kırpma Algılayıcı
EAR (Eye Aspect Ratio) kullanarak göz kırpmasını algılar
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Tuple
import time
import numpy as np


class BlinkType(Enum):
    """Göz kırpma türleri"""
    LEFT = "left"
    RIGHT = "right"
    BOTH = "both"


@dataclass
class BlinkEvent:
    """Göz kırpma olayı"""
    blink_type: BlinkType
    timestamp: float
    duration: float = 0.0


class BlinkDetector:
    """
    EAR tabanlı göz kırpma algılayıcı
    
    EAR (Eye Aspect Ratio):
    - Göz açıkken ~0.25-0.30
    - Göz kapalıyken ~0.10 veya daha düşük
    """
    
    # MediaPipe Face Mesh göz landmark indeksleri
    LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]  # Sağ-üst, sol-üst, sol-alt, sağ-alt
    RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    
    def __init__(
        self,
        ear_threshold: float = 0.21,
        consecutive_frames: int = 3,
        cooldown: float = 0.5
    ):
        self.ear_threshold = ear_threshold
        self.consecutive_frames = consecutive_frames
        self.cooldown = cooldown
        
        # Göz kapalı frame sayaçları
        self.left_closed_frames = 0
        self.right_closed_frames = 0
        
        # Son kırpma zamanları
        self.last_left_blink = 0.0
        self.last_right_blink = 0.0
        self.last_both_blink = 0.0
        
        # EAR değerleri
        self.left_ear = 0.3
        self.right_ear = 0.3
        
    def update_thresholds(self, ear_threshold: float, cooldown: float):
        """Eşik değerlerini güncelle"""
        self.ear_threshold = ear_threshold
        self.cooldown = cooldown
        
    def _calculate_ear(self, eye_landmarks: List[Tuple[float, float]]) -> float:
        """
        Eye Aspect Ratio hesapla
        
        EAR = (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|)
        
        p1, p4: Yatay noktalar (göz köşeleri)
        p2, p3, p5, p6: Dikey noktalar (göz kapakları)
        """
        if len(eye_landmarks) < 6:
            return 0.3
            
        p1, p2, p3, p4, p5, p6 = eye_landmarks[:6]
        
        # Dikey mesafeler
        v1 = np.linalg.norm(np.array(p2) - np.array(p6))
        v2 = np.linalg.norm(np.array(p3) - np.array(p5))
        
        # Yatay mesafe
        h = np.linalg.norm(np.array(p1) - np.array(p4))
        
        if h == 0:
            return 0.3
            
        ear = (v1 + v2) / (2.0 * h)
        return ear
        
    def detect(
        self,
        left_eye_landmarks: Optional[List[Tuple[float, float]]],
        right_eye_landmarks: Optional[List[Tuple[float, float]]]
    ) -> Optional[BlinkEvent]:
        """
        Göz kırpması algıla
        
        Returns:
            BlinkEvent veya None
        """
        now = time.time()
        
        # EAR hesapla
        if left_eye_landmarks:
            self.left_ear = self._calculate_ear(left_eye_landmarks)
        if right_eye_landmarks:
            self.right_ear = self._calculate_ear(right_eye_landmarks)
            
        # Göz kapalı mı kontrol et
        left_closed = self.left_ear < self.ear_threshold
        right_closed = self.right_ear < self.ear_threshold
        
        # Frame sayaçlarını güncelle
        if left_closed:
            self.left_closed_frames += 1
        else:
            # Sol göz açıldı - kırpma algıla
            if self.left_closed_frames >= self.consecutive_frames:
                if now - self.last_left_blink > self.cooldown:
                    # Sağ göz de aynı anda kapalıydı mı?
                    if self.right_closed_frames >= self.consecutive_frames:
                        if now - self.last_both_blink > self.cooldown:
                            self.last_both_blink = now
                            self.last_left_blink = now
                            self.last_right_blink = now
                            self.left_closed_frames = 0
                            self.right_closed_frames = 0
                            return BlinkEvent(BlinkType.BOTH, now)
                    else:
                        self.last_left_blink = now
                        self.left_closed_frames = 0
                        return BlinkEvent(BlinkType.LEFT, now)
            self.left_closed_frames = 0
            
        if right_closed:
            self.right_closed_frames += 1
        else:
            # Sağ göz açıldı - kırpma algıla
            if self.right_closed_frames >= self.consecutive_frames:
                if now - self.last_right_blink > self.cooldown:
                    # Sol göz de aynı anda kapalıydı mı?
                    if self.left_closed_frames >= self.consecutive_frames:
                        if now - self.last_both_blink > self.cooldown:
                            self.last_both_blink = now
                            self.last_left_blink = now
                            self.last_right_blink = now
                            self.left_closed_frames = 0
                            self.right_closed_frames = 0
                            return BlinkEvent(BlinkType.BOTH, now)
                    else:
                        self.last_right_blink = now
                        self.right_closed_frames = 0
                        return BlinkEvent(BlinkType.RIGHT, now)
            self.right_closed_frames = 0
            
        return None
        
    def get_ear_values(self) -> Tuple[float, float]:
        """Güncel EAR değerlerini döndür (sol, sağ)"""
        return (self.left_ear, self.right_ear)
