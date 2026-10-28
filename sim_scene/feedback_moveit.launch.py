import sys
import os
import json
from pathlib import Path
sys.path.insert(0, '/experiment/src')
from launch import LaunchDescription
from launch_ros.actions import Node
from modules.moveit_controller import build_ur5_moveit_config


def generate_launch_description():
    cfg = build_ur5_moveit_config()
    cfg['use_sim_time'] = False
    runtime_path=Path('/experiment/sim/runtime_config.json')
    runtime=json.loads(runtime_path.read_text()) if runtime_path.exists() else {}
    embedded=os.environ.get('EMBEDDED_TRAJECTORY',str(int(runtime.get('embedded_trajectory',True)))) == '1'
    return LaunchDescription([
        Node(package='robot_state_publisher', executable='robot_state_publisher',
             parameters=[{'robot_description': cfg['robot_description'], 'use_sim_time': False}]),
        Node(package='moveit_ros_move_group', executable='move_group', parameters=[cfg]),
        *([] if embedded else
          [Node(executable='python3', arguments=['/experiment/src/sim_scene/feedback_converter.py'])]),
    ])
