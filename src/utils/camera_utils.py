
import cv2

def get_available_cameras(max_check=5):
    """
    Check the first few camera indexes to see which ones are available.
    Returns a list of tuples: (index, name)
    """
    available_cameras = []
    
    for i in range(max_check):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Try to read a frame
            ret, _ = cap.read()
            if ret:
                available_cameras.append((i, f"Camera {i}"))
            cap.release()
            
    return available_cameras
