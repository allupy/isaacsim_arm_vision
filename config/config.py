# config/config.py
"""配置模块:DetectorConfig 从 YAML 加载,附带权重路径解析"""
import os
import yaml
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# 项目根目录(config/ 的上一级)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 默认相机内参(D455,1280x720,与手眼标定时一致)
DEFAULT_CAMERA_MATRIX = [
    [634.165821427196, 0.0, 636.8610491500791],
    [0.0, 634.0985503275863, 357.7665813647669],
    [0.0, 0.0, 1.0],
]
DEFAULT_DIST_COEFFS = [
    0.0028141594696008986, -0.006156795039937514,
    -0.0006492150364271804, -0.0012946439050663806, 0.003993861075015315,
]


def resolve_weight_path(path: str) -> str:
    """按  当前路径 → 项目根 → ../vision 的顺序解析权重文件,找到即返回绝对路径"""
    if os.path.isabs(path) and os.path.exists(path):
        return path
    candidates = [
        path,
        os.path.join(PROJECT_ROOT, path),
        os.path.join(PROJECT_ROOT, os.pardir, 'vision', os.path.basename(path)),
    ]
    for cand in candidates:
        if os.path.exists(cand):
            return os.path.abspath(cand)
    return path  # 找不到时原样返回,让上层报出明确的加载错误


@dataclass
class DetectorConfig:
    """检测器配置(检测 + 分割 + 相机 + 机械臂)"""
    save_dir: str = "detection_results"
    default_classes: List[str] = field(default_factory=lambda: ["cube"])
    model_path: str = "yolov8s-worldv2.pt"
    confidence_threshold: float = 0.02
    max_detections: int = 20
    display_stats: bool = True
    auto_save: bool = True

    # SAM 分割
    sam_checkpoint: str = "sam_vit_h_4b8939.pth"
    sam_model_type: str = "vit_h"

    # 相机内参
    camera_matrix: Optional[np.ndarray] = None
    dist_coeffs: Optional[np.ndarray] = None

    # 机械臂参数(传给 ArmControllerWithGripper)
    arm: dict = field(default_factory=dict)

    # 抓取是否使用深度估计的 z(默认 False:抓取平面高度取 0,与原有行为一致)
    use_estimated_z: bool = False

    def __post_init__(self):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(DEFAULT_CAMERA_MATRIX)
        if self.dist_coeffs is None:
            self.dist_coeffs = np.array(DEFAULT_DIST_COEFFS)

    @classmethod
    def from_yaml(cls, yaml_path: str = None):
        """从YAML文件加载配置"""
        config = cls()

        if yaml_path and os.path.exists(yaml_path):
            with open(yaml_path, 'r') as f:
                yaml_data = yaml.safe_load(f) or {}

                # 相机内参(嵌套列表 → ndarray)
                if 'camera_matrix' in yaml_data:
                    config.camera_matrix = np.array(yaml_data['camera_matrix'], dtype=float)
                    yaml_data.pop('camera_matrix')
                if 'dist_coeffs' in yaml_data:
                    config.dist_coeffs = np.array(yaml_data['dist_coeffs'], dtype=float)
                    yaml_data.pop('dist_coeffs')

                # 相机→机械手外参
                if 'R_cam2gripper' in yaml_data:
                    config.R_cam2gripper = np.array(yaml_data['R_cam2gripper'])
                    yaml_data.pop('R_cam2gripper')
                if 't_cam2gripper' in yaml_data:
                    config.t_cam2gripper = np.array(yaml_data['t_cam2gripper'])
                    yaml_data.pop('t_cam2gripper')

                # 机械臂参数段
                if 'arm' in yaml_data:
                    config.arm = dict(yaml_data['arm'])
                    yaml_data.pop('arm')

                # 其余标量字段
                for key, value in yaml_data.items():
                    if hasattr(config, key) and not key.startswith('_'):
                        setattr(config, key, value)

        # 权重路径解析(相对路径 → 绝对路径)
        config.model_path = resolve_weight_path(config.model_path)
        config.sam_checkpoint = resolve_weight_path(config.sam_checkpoint)

        return config

    def to_dict(self):
        """转换为字典(用于YAML保存),numpy数组转列表"""
        data = {}
        for key, val in self.__dict__.items():
            if key.startswith('_'):
                continue
            if isinstance(val, np.ndarray):
                data[key] = val.tolist()
            else:
                data[key] = val
        return data

    def to_yaml(self, yaml_path: str):
        """保存配置到YAML"""
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)

    def get_cam2gripper_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取相机到机械手的位姿变换 (R, t)"""
        if getattr(self, 'R_cam2gripper', None) is None or getattr(self, 't_cam2gripper', None) is None:
            raise ValueError("Camera to gripper pose not configured")
        return self.R_cam2gripper, self.t_cam2gripper
