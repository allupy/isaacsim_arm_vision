# modules/class_manager.py
from typing import List, Callable, Optional
import numpy as np

class ClassManager:
    """
    类别管理器
    也就是文本交互，伪聊天模块

    """
    
    def __init__(self, default_classes: List[str]):
        self.default_classes = default_classes.copy()
        self.current_classes = default_classes.copy()
        self.on_classes_changed: Optional[Callable] = None
    
    def get_classes(self) -> List[str]:
        """获取当前类别"""
        return self.current_classes.copy()
    
    def set_classes(self, classes: List[str], notify: bool = True):
        """设置类别"""
        # 去重
        unique_classes = list(dict.fromkeys(classes))
        
        # 确保有空字符串作为背景类
        if "" not in unique_classes:
            unique_classes.append("")
        
        self.current_classes = unique_classes
        
        if notify and self.on_classes_changed:
            self.on_classes_changed(self.current_classes)
    
    def reset_to_default(self, notify: bool = True):
        """重置为默认类别"""
        self.current_classes = self.default_classes.copy()
        
        if notify and self.on_classes_changed:
            self.on_classes_changed(self.current_classes)
    
    def parse_input(self, input_str: str) -> Optional[List[str]]:
        """
        解析用户输入的类别字符串
        
        Returns:
            List[str] or None: 解析后的类别列表，无效输入返回None
        """
        input_str = input_str.strip()
        
        if not input_str or input_str.lower() == 'cancel':
            return None
        
        if input_str.lower() == 'reset':
            return []
        
        # 分割并清理
        classes = [cls.strip() for cls in input_str.split(',') if cls.strip()]
        
        if not classes:
            return None
        
        return classes
    
    def format_classes_display(self) -> str:
        """格式化显示类别"""
        valid_classes = [c for c in self.current_classes if c]
        if not valid_classes:
            return "（无检测类别）"
        return ", ".join(valid_classes)