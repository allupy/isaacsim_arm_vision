# config/config.py
import yaml
import os
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

@dataclass
class DetectorConfig:
    """检测器配置"""
    save_dir: str = "detection_results"
    default_classes: List[str] = field(default_factory=lambda: ["bright blue-purple cube in the lower right corner", "cube", ""])
    model_path: str = "yolov8s-world.pt"
    confidence_threshold: float = 0.1
    max_detections: int = 20
    display_stats: bool = True
    auto_save: bool = True
    
    # 相机到机械手位姿参数
    R_cam2gripper: Optional[np.ndarray] = None
    t_cam2gripper: Optional[np.ndarray] = None
    
    @classmethod
    def from_yaml(cls, yaml_path: str = None):
        """从YAML文件加载配置
        
        Args:
            yaml_path: YAML配置文件路径
            
        Returns:
            DetectorConfig实例
        """
        config = cls()
        
        if yaml_path and os.path.exists(yaml_path):
            with open(yaml_path, 'r') as f:
                yaml_data = yaml.safe_load(f)
                
                # 处理相机位姿参数
                if 'R_cam2gripper' in yaml_data:
                    config.R_cam2gripper = np.array(yaml_data['R_cam2gripper'])
                if 't_cam2gripper' in yaml_data:
                    config.t_cam2gripper = np.array(yaml_data['t_cam2gripper'])
                
                # 设置其他属性
                for key, value in yaml_data.items():
                    if hasattr(config, key) and key not in ['R_cam2gripper', 't_cam2gripper']:
                        setattr(config, key, value)
                        
        return config
    
    def to_dict(self):
        """转换为字典，用于YAML保存
        
        注意：numpy数组会被转换为列表
        """
        data = {}
        for key, val in self.__dict__.items():
            if not key.startswith('_'):
                if isinstance(val, np.ndarray):
                    data[key] = val.tolist()
                else:
                    data[key] = val
        return data
    
    def to_yaml(self, yaml_path: str):
        """保存配置到YAML文件
        
        Args:
            yaml_path: 保存路径
        """
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    def get_cam2gripper_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取相机到机械手的位姿变换
        
        Returns:
            Tuple[R, t]: 旋转矩阵和平移向量
        """
        if self.R_cam2gripper is None or self.t_cam2gripper is None:
            raise ValueError("Camera to gripper pose not configured")
        return self.R_cam2gripper, self.t_cam2gripper