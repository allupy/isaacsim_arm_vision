# modules/display_manager.py
import cv2
from typing import Optional
import numpy as np

class DisplayManager:
    """显示管理器"""
    
    def __init__(self, window_name: str = "Detection"):
        self.window_name = window_name
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    def show(self, image: Optional[np.ndarray], delay: int = 1):
        """显示图像"""
        if image is not None:
            cv2.imshow(self.window_name, image)
        
        key = cv2.waitKey(delay) & 0xFF
        return key
    
    def close(self):
        """关闭窗口"""
        cv2.destroyWindow(self.window_name)
    
    def is_open(self) -> bool:
        """检查窗口是否打开"""
        try:
            return cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) >= 1
        except:
            return False