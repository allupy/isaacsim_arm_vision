"""CLI/configuration for the legacy interactive entry, importable without ROS."""
import argparse
from pathlib import Path
from config.config import DetectorConfig, PROJECT_ROOT


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='视觉机械臂抓取(Isaac Sim 版)')
    parser.add_argument('--once', action='store_true', help='无头单次运行')
    parser.add_argument('--observe-only', action='store_true', help='仅检测，不初始化机械臂控制器，不发送运动指令')
    parser.add_argument('--config', default=str(Path(PROJECT_ROOT) / 'config/default_config.yaml'))
    parser.add_argument('--classes', nargs='+', help='英文目标类别，例如 --classes apple banana cup')
    parser.add_argument('--confidence', type=float, help='检测阈值；低阈值可能增加误识别')
    parser.add_argument('--rgb-topic', default='/rgb')
    parser.add_argument('--depth-topic', default='/depth')
    args = parser.parse_args(argv)
    if args.confidence is not None and not 0 < args.confidence <= 1:
        parser.error('--confidence must be in (0, 1]')
    return args


def load_config(args):
    config = DetectorConfig.from_yaml(args.config)
    if args.classes:
        config.default_classes = list(dict.fromkeys(name.strip() for name in args.classes if name.strip()))
        if not config.default_classes:
            raise ValueError('At least one nonempty class is required')
    if args.confidence is not None:
        config.confidence_threshold = args.confidence
    return config
