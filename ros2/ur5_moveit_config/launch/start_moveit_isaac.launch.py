#!/usr/bin/env python3
#start_moveit_isaac.launch.py
#!/usr/bin/env python3
from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_demo_launch
from launch import LaunchDescription
from launch_ros.actions import Node
import os

from ament_index_python.packages import get_package_share_directory

# def generate_launch_description():
#     # 获取MoveIt配置
#     moveit_config = (
#             MoveItConfigsBuilder(
#                 "ur5",
#                 package_name="ur5_moveit_config"
#             )
#             .planning_pipelines(
#                 pipelines=["ompl"]  # 明确指定OMPL规划管道
#             )
#             .robot_description_semantic(file_path="config/ur5.srdf")
#             .robot_description_kinematics(file_path="config/kinematics.yaml")
#             .trajectory_execution(file_path="config/moveit_controllers.yaml")
#             .planning_scene_monitor(
#                 publish_planning_scene=True,
#                 publish_geometry_updates=True,
#                 publish_state_updates=True,
#                 publish_transforms_updates=True,
#                 publish_robot_description=True,
#                 publish_robot_description_semantic=True
#             )
#             .to_moveit_configs()
#         )

#     kinematics_config_path = os.path.join(
#         get_package_share_directory('ur5_moveit_config'),
#         'config',
#         'kinematics.yaml'
#     )

#     # 4. 禁用Octomap（从错误信息看，你不需要3D传感器）
#     moveit_config_dict = moveit_config.to_dict()
#     moveit_config_dict['planning_scene_monitor'] = {
#         'publish_planning_scene': True,
#         'publish_geometry_updates': True,
#         'publish_state_updates': True,
#         'publish_transforms_updates': True,
#         'publish_robot_description': True,
#         'publish_robot_description_semantic': True,
#         'publish_sensors': False,  # 关键：不发布传感器
#     }
#         # 确保没有启用任何传感器
#     moveit_config_dict['sensors'] = {}

#     # 关键：确保planning_pipeline参数存在
#     # 关键修改3：确保planning_pipeline参数正确
#     moveit_config_dict['move_group'] = {
#         'planning_plugin': 'ompl_interface/OMPLPlanner',
#         'request_adapters': """default_planner_request_adapters/AddTimeOptimalParameterization default_planner_request_adapters/FixWorkspaceBounds default_planner_request_adapters/FixStartStateBounds default_planner_request_adapters/FixStartStateCollision default_planner_request_adapters/FixStartStatePathConstraints""",
#         'start_state_max_bounds_error': 0.1,
#         'planning_pipelines': ['ompl'],
#         'default_planning_pipeline': 'ompl'
#     }

#     # 关键修改4：添加OMPL配置
#     if 'ompl' not in moveit_config_dict:
#         moveit_config_dict['ompl'] = {}

#     moveit_config_dict['ompl'].update({
#         'planning_plugin': 'ompl_interface/OMPLPlanner',
#         'request_adapters': """default_planner_request_adapters/AddTimeOptimalParameterization default_planner_request_adapters/FixWorkspaceBounds default_planner_request_adapters/FixStartStateBounds default_planner_request_adapters/FixStartStateCollision default_planner_request_adapters/FixStartStatePathConstraints""",
#         'start_state_max_bounds_error': 0.1
#     })

#     # 关键修改5：禁用octomap更新
#     moveit_config_dict.update({
#         'octomap_enabled': False,
#         'publish_octomap': False
#     })

#     # 生成demo启动
#     demo_launch = generate_demo_launch(moveit_config)

#     # 添加转换节点
#     converter_node = Node(
#         executable='python3',
#         arguments=[
#             os.path.join(os.path.dirname(__file__), 'trajectory_converter.py'),
#             '--ros-args',
#             '-p', 'use_sim_time:=true',
#         ],
#         name='trajectory_converter',
#         output='screen',
#     )

#     move_group_node = Node(
#         package='moveit_ros_move_group',
#         executable='move_group',
#         output='screen',
#         parameters=[moveit_config_dict],
#     )

#     # 返回组合的launch
#     launch_desc = LaunchDescription()
#     launch_desc.add_action(demo_launch)
#     launch_desc.add_action(converter_node)
#     launch_desc.add_action(move_group_node)

#     return launch_desc
def generate_launch_description():
    # 获取包路径
    package_path = get_package_share_directory('ur5_moveit_config')

    # 构建MoveIt配置 - 按照Issue的解决方案
    moveit_config = (
        MoveItConfigsBuilder(
            robot_name="ur5",
            package_name="ur5_moveit_config"
        )
        .robot_description(file_path="config/ur5.urdf.xacro")
        .robot_description_semantic(file_path="config/ur5.srdf")
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .planning_pipelines(
            pipelines=["ompl"],
            default_planning_pipeline="ompl"
        )
        .sensors_3d(file_path="config/sensors_3d.yaml")
        .moveit_cpp(  # 关键：添加moveit_cpp配置
            file_path=os.path.join(package_path, "config", "moveit_cpp.yaml")
        )
        .to_moveit_configs()
    )

    # 获取配置
    config_dict = moveit_config.to_dict()

    # 调试：打印规划管道配置
    print("=" * 80)
    if 'planning_pipelines' in config_dict:
        print("✅ planning_pipelines配置存在:", config_dict['planning_pipelines'])
    else:
        print("❌ planning_pipelines配置不存在")
        # 手动添加
        config_dict['planning_pipelines'] = {
            'pipeline_names': ['ompl'],
            'default_planning_pipeline': 'ompl'
        }
    print("=" * 80)

    # 创建move_group节点
    move_group_node = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        output='screen',
        parameters=[config_dict],
        name='move_group',
        arguments=['--ros-args', '--log-level', 'info']
    )

    # 如果需要转换节点
    converter_node = Node(
        executable='python3',
        arguments=[
            os.path.join(os.path.dirname(__file__), 'trajectory_converter.py'),
            '--ros-args',
            '-p', 'use_sim_time:=true',
        ],
        name='trajectory_converter',
        output='screen',
    )

    return LaunchDescription([
        move_group_node,
        converter_node
    ])
