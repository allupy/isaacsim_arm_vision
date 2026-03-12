# modules/keyboard_handler.py
import sys
import select
import termios
import tty
import atexit
from typing import Callable, Optional
import numpy as np

class KeyboardHandler:
    """非阻塞键盘输入处理器"""
    
    def __init__(self):
        self.old_settings = None
        self._setup_nonblocking()
        atexit.register(self.restore)
    
    def _setup_nonblocking(self):
        """设置非阻塞输入"""
        self.old_settings = termios.tcgetattr(sys.stdin)
        tty.setraw(sys.stdin.fileno())
    
    def restore(self):
        """恢复终端设置"""
        if self.old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
    
    def get_key(self, timeout: float = 0.1) -> Optional[str]:
        """
        非阻塞获取按键
        
        Args:
            timeout: 超时时间（秒）
            
        Returns:
            str or None: 按键字符，无输入时返回None
        """
        try:
            if sys.stdin in select.select([sys.stdin], [], [], timeout)[0]:
                return sys.stdin.read(1).lower()
        except (IOError, ValueError):
            pass
        return None
    
    def get_blocking_input(self, prompt: str = "") -> str:
        """
        阻塞模式获取用户输入
        
        Args:
            prompt: 提示信息
            
        Returns:
            str: 用户输入
        """
        self.restore()  # 恢复阻塞模式
        try:
            return input(prompt)
        finally:
            self._setup_nonblocking()  # 重新设置非阻塞