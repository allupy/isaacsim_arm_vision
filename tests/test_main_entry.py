"""Regressions for the configuration used by main.py, without ROS or motion."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.config import DetectorConfig
from config.main_entry import parse_args, load_config
from modules.class_manager import ClassManager


class MainEntryTests(unittest.TestCase):
    def test_default_yaml_does_not_restore_old_model_or_threshold(self):
        args = parse_args(['--once', '--observe-only'])
        with tempfile.TemporaryDirectory() as tmp:
            previous = os.getcwd()
            try:
                os.chdir(tmp)
                config = load_config(args)
            finally:
                os.chdir(previous)
        self.assertEqual(Path(config.model_path).name, 'yolov8s-worldv2.pt')
        self.assertEqual(config.confidence_threshold, .02)
        self.assertEqual(config.default_classes, ['red cube', 'green cube', 'blue cube'])
        self.assertEqual(config.sam_model_type, 'vit_b')

    def test_classes_and_threshold_can_match_the_current_scene(self):
        config = load_config(parse_args(['--classes', 'apple', 'banana', 'cup', 'apple', '--confidence', '.08']))
        self.assertEqual(config.default_classes, ['apple', 'banana', 'cup'])
        self.assertEqual(config.confidence_threshold, .08)

    def test_explicit_missing_config_is_not_silently_ignored(self):
        with self.assertRaises(FileNotFoundError):
            DetectorConfig.from_yaml('/missing/arm-config-file.yaml')

    def test_runtime_weight_override_is_resolved(self):
        with tempfile.TemporaryDirectory() as tmp:
            weight = Path(tmp) / 'custom.pt'
            weight.touch()
            with patch.dict(os.environ, ROBOT_YOLO_WEIGHTS=str(weight)):
                self.assertEqual(load_config(parse_args([])).model_path, str(weight))

    def test_interactive_changes_do_not_add_an_empty_background_prompt(self):
        manager = ClassManager(['cube'])
        manager.set_classes(['apple', 'cup', ''])
        self.assertEqual(manager.get_classes(), ['apple', 'cup'])
        with self.assertRaises(ValueError):
            manager.set_classes([' ', ''])


if __name__ == '__main__':
    unittest.main()
