#!/usr/bin/env python3
"""
视觉机械臂抓取主程序(Isaac Sim 版)

流程:订阅 /rgb /depth → YOLO-World 检测 + SAM 分割(最优目标)
     → 掩码质心 + 深度 → 相机系 3D → 手眼外参 → 基座系坐标
     → 机械臂抓取放置。
显示在独立进程中运行(cv2.imshow 与 ROS 回调分离)。
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import time
import tf2_ros
import cv2
import multiprocessing as mp

from modules.detection_manager import DetectionManager
from modules.keyboard_handler import KeyboardHandler
from modules.io_manager import IOManager
from modules.class_manager import ClassManager
from modules.utils import (quaternion_to_euler, build_transform_from_matrix,
                           build_transform, quaternion_to_matrix)
from modules.arm_manager import ArmControllerWithGripper
from config.config import DetectorConfig

# 实时帧发送到显示进程的节流间隔(秒)
LIVE_SEND_INTERVAL = 1.0 / 15


def display_process(image_queue, exit_event, window_name="Detection"):
    """独立显示进程:负责 cv2.imshow 与键盘事件(ESC退出)"""
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    print(f"📺 显示进程已启动 (PID: {mp.current_process().pid})")

    while not exit_event.is_set():
        try:
            if not image_queue.empty():
                img = image_queue.get_nowait()
                cv2.imshow(window_name, img)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                exit_event.set()
        except Exception as e:
            print(f"⚠️ 显示进程出错: {e}")
            continue

    cv2.destroyAllWindows()
    print("📺 显示进程已退出")


class EnhancedDetectorNode(Node):
    """ROS2 检测 + 抓取主节点"""

    def __init__(self, config: DetectorConfig):
        super().__init__('enhanced_detector')

        self.config = config
        self.bridge = CvBridge()

        # TF(末端执行器位姿)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.ee_frame = 'wrist_3_link'
        self.base_frame = 'base_link'
        self.R_cam2gripper, self.t_cam2gripper = config.get_cam2gripper_pose()

        # 相机内参(来自配置,默认 D455 1280x720)
        self.camera_matrix = config.camera_matrix
        self.dist_coeffs = config.dist_coeffs

        # 最新帧缓存
        self.current_rgb = None
        self.current_depth = None
        self.current_robot_pose = None

        # 检测结果显示状态(结果帧只发送一次,窗口会保持最后一帧)
        self.show_result = False
        self.result_img = None
        self.result_show_start = 0.0
        self.result_show_duration = 3.0
        self._result_sent = False
        self._last_live_send = 0.0

        # 显示进程(spawn 方式与 ROS2 兼容)
        self.mp_context = mp.get_context('spawn')
        self.image_queue = self.mp_context.Queue(maxsize=5)
        self.exit_event = self.mp_context.Event()
        self.display_proc = self.mp_context.Process(
            target=display_process,
            args=(self.image_queue, self.exit_event, "Detection")
        )
        self.display_proc.start()

        # 模块初始化
        self.io_manager = IOManager(config.save_dir)
        self.keyboard_handler = KeyboardHandler()
        self.arm_controller = ArmControllerWithGripper(self, params=config.arm)

        self.class_manager = ClassManager(config.default_classes)
        self.class_manager.on_classes_changed = self._on_classes_changed

        self.detection_manager = DetectionManager(config)

        self._setup_subscribers()
        self.running = True
        self._print_welcome()

    def _setup_subscribers(self):
        """订阅 RGB 与深度(取最新帧;时间同步是已知待改进项,见 README)"""
        self.rgb_sub = self.create_subscription(Image, '/rgb', self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/depth', self.depth_callback, 10)
        self.get_logger().info("已启用RGB-D订阅（无时间同步）")

    def rgb_callback(self, msg):
        try:
            self.current_rgb = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"RGB回调失败: {e}")

    def depth_callback(self, msg):
        try:
            self.current_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().error(f"深度回调失败: {e}")

    # ------------------------------------------------------------------
    # 显示
    # ------------------------------------------------------------------
    def send_to_display(self, img):
        """发送图像到显示进程(队列满则丢弃,不阻塞主循环)"""
        try:
            if not self.image_queue.full():
                self.image_queue.put_nowait(img.copy())
            return True
        except Exception as e:
            self.get_logger().warn(f"发送图像到显示进程失败: {e}")
            return False

    def _manage_display(self):
        """结果帧只发一次;实时帧按 LIVE_SEND_INTERVAL 节流发送"""
        current_time = time.time()

        if self.show_result and self.result_img is not None:
            if current_time - self.result_show_start < self.result_show_duration:
                if not self._result_sent:
                    self.send_to_display(self.result_img)
                    self._result_sent = True
            else:
                # 超时切回实时帧
                self.show_result = False
                self.result_img = None
                self._result_sent = False
        elif self.current_rgb is not None:
            if current_time - self._last_live_send >= LIVE_SEND_INTERVAL:
                if self.send_to_display(self.current_rgb):
                    self._last_live_send = current_time

    # ------------------------------------------------------------------
    # 帧与位姿获取
    # ------------------------------------------------------------------
    def get_current_frames(self):
        """
        获取当前 RGB、深度与末端位姿。
        位姿获取失败时回退到上一次成功值(而非未定义变量)。
        """
        rgb = self.current_rgb.copy() if self.current_rgb is not None else None
        depth = self.current_depth.copy() if self.current_depth is not None else None

        pose = None
        success, robot_pose = self.get_robot_pose()
        if success and robot_pose is not None:
            self.current_robot_pose = robot_pose
            pose = robot_pose
        elif self.current_robot_pose is not None:
            pose = self.current_robot_pose

        return rgb, depth, pose

    def get_robot_pose(self):
        """获取末端执行器在基座系下的位姿(TF)"""
        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame, self.ee_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
            t = transform.transform.translation
            r = transform.transform.rotation
            R_ee2base = quaternion_to_matrix([r.x, r.y, r.z, r.w])
            return True, {
                'position': [float(t.x), float(t.y), float(t.z)],
                'orientation_quat': [float(r.x), float(r.y), float(r.z), float(r.w)],
                'rotation_matrix': R_ee2base,
                'base_frame': self.base_frame,
                'ee_frame': self.ee_frame
            }
        except Exception as e:
            self.get_logger().error(f"获取机器人位姿失败: {e}")
            return False, None

    # ------------------------------------------------------------------
    # 位姿估计
    # ------------------------------------------------------------------
    def pose_estimation(self, rgb, depth, ee_in_base, mask):
        """掩码质心 + 深度 → 相机系 3D 点 → 基座系坐标"""
        camera_matrix = self.camera_matrix

        # 1. 掩码质心像素坐标
        M = cv2.moments(mask)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        # 2. 深度
        Z = depth[cy, cx]

        # 3. 像素 → 相机系
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx_cam = camera_matrix[0, 2]
        cy_cam = camera_matrix[1, 2]
        X_cam = (cx - cx_cam) * Z / fx
        Y_cam = (cy - cy_cam) * Z / fy

        point_camera = np.array([X_cam, Y_cam, Z, 1.0]).reshape(4, 1)

        # 4. 相机系 → 基座系:T_base_ee @ T_ee_cam @ p_cam
        R_ee2base = ee_in_base['rotation_matrix']
        t_ee2base = np.array(ee_in_base['position']).flatten()
        T_ee2base = build_transform_from_matrix(t_ee2base, R_ee2base)

        t_cam2ee = self.t_cam2gripper.flatten()
        T_cam2ee = build_transform_from_matrix(t_cam2ee, self.R_cam2gripper)

        T_cam2base = T_ee2base @ T_cam2ee
        point_base = (T_cam2base @ point_camera)[:3].flatten()

        return tuple(point_base)

    # ------------------------------------------------------------------
    # 命令处理
    # ------------------------------------------------------------------
    def _print_welcome(self):
        print("\n" + "=" * 60)
        print("🎯 视觉机械臂抓取(Isaac Sim 版)")
        print("=" * 60)
        print(f"\n当前检测类别: {self.class_manager.format_classes_display()}")
        print(f"保存目录: {self.io_manager.save_dir.absolute()}")
        print(f"显示进程PID: {self.display_proc.pid}")
        self._print_help()

    def _print_help(self):
        print("\n📋 快捷键:")
        print("  [d] 实时检测当前帧并抓取最优目标")
        print("  [s] 拍照并保存检测结果")
        print("  [c] 修改检测类别")
        print("  [r] 机械臂归位")
        print("  [l] 显示历史记录")
        print("  [h] 显示此帮助")
        print("  [q] 退出程序")
        print("  [ESC] 关闭显示窗口")
        print("=" * 60)

    def _on_classes_changed(self, classes: list):
        self.detection_manager.set_classes(classes)
        print(f"\n✅ 检测类别已更新: {self.class_manager.format_classes_display()}")

    def process_command(self, command: str):
        if command == 'p':
            success, robot_pose = self.get_robot_pose()
            print('机械臂位置获取延迟测试')
            print(f'机械臂位置: {robot_pose}')
        elif command == 'd':
            self._command_detect()
        elif command == 's':
            self._command_save()
        elif command == 'c':
            self._command_change_classes()
        elif command == 'r':
            status = self.arm_controller.get_status()
            print(status)
            self.arm_controller.go_home()
        elif command == 'l':
            self._command_show_history()
        elif command == 'h':
            self._print_help()
        elif command == 'q':
            self.running = False
            print("\n🛑 正在退出...")
        else:
            print(f"\n❌ 未知命令: {command}，输入 'h' 查看帮助")

    def _command_detect(self):
        """实时检测 + 抓取置信度最高的目标"""
        rgb, depth, robot_pose = self.get_current_frames()
        if rgb is None:
            print("❌ 错误: 没有可用的图像帧")
            return
        if depth is None:
            print("❌ 错误: 没有可用的深度帧")
            return
        if robot_pose is None:
            print("❌ 错误: 未能获取机械臂位姿(TF不可用),无法做位姿估计")
            return

        print("\n🎯 运行实时检测...")
        result_img, detections = self.detection_manager.detect(rgb)

        if result_img is not None:
            self.result_img = result_img.copy()
            self.show_result = True
            self.result_show_start = time.time()
            self._result_sent = False
            print(f"✅ 检测完成，找到 {len(detections)} 个目标（将显示{self.result_show_duration}秒）")

        if len(detections) == 0:
            return

        best_detection = max(detections, key=lambda d: d.confidence)
        mask = best_detection.mask

        if mask is None or not mask.any():
            print(f"警告: 检测到{best_detection.class_name}但没有掩码，无法进行姿态估计")
            return

        try:
            x, y, z = self.pose_estimation(rgb, depth, robot_pose, mask)
            print(f"物体位置在({x:.3f},{y:.3f},{z:.3f})")

            # 抓取 z:默认取 0(桌面高度);use_estimated_z=True 时用深度估计值
            grasp_z = z if self.config.use_estimated_z else 0.0

            if self.arm_controller.execute_pick_place(
                [x, y, grasp_z],
                place_position=[0.0, -0.5, 0.10],
            ):
                print("✅ 抓取完成")
            else:
                print("❌ 抓取失败")
        except ZeroDivisionError as e:
            print(f"计算姿态时出错: {e}，掩码可能有问题")
        except Exception as e:
            print(f"姿态估计失败: {e}")

    def _command_save(self):
        """拍照并保存检测结果"""
        rgb = self.current_rgb
        if rgb is None:
            print("❌ 错误: 没有可用的图像帧")
            return

        print("\n📸 拍照并保存检测结果...")
        result_img, detections = self.detection_manager.detect(rgb)

        if result_img is not None:
            self.result_img = result_img.copy()
            self.show_result = True
            self.result_show_start = time.time()
            self._result_sent = False

            saved_files = self.io_manager.save_detection_result(
                raw_image=rgb,
                result_image=result_img,
                detections=detections,
                classes=self.class_manager.get_classes()
            )
            print(f"\n💾 检测结果已保存:")
            print(f"  原始图像: {saved_files['raw']}")
            print(f"  结果图像: {saved_files['result']}")
            print(f"  检测信息: {saved_files['info']}")
            print(f"  JSON数据: {saved_files['json']}")

    def _command_change_classes(self):
        print("\n" + "=" * 60)
        print("📝 修改检测类别")
        print("=" * 60)
        print(f"当前类别: {self.class_manager.format_classes_display()}")
        print("\n💡 输入新的检测类别（用逗号分隔，输入 'reset' 重置，'cancel' 取消）")
        print("示例: person,car,dog,cat")
        print("-" * 40)

        user_input = self.keyboard_handler.get_blocking_input("\n请输入新类别: ")
        if user_input is None:
            return

        new_classes = self.class_manager.parse_input(user_input)
        if new_classes is None:
            print("取消修改")
        elif new_classes == []:
            self.class_manager.reset_to_default()
        else:
            self.class_manager.set_classes(new_classes)

    def _command_show_history(self):
        print("\n" + "=" * 60)
        print("📊 检测历史")
        print("=" * 60)

        history = self.io_manager.load_history(limit=5)
        if not history:
            print("暂无检测记录")
            return

        print(f"找到 {len(history)} 条最近的检测记录:\n")
        for i, record in enumerate(history, 1):
            print(f"{i}. 时间: {record['timestamp']}")
            print(f"   检测数: {record['detections_count']}")
            print(f"   类别: {', '.join([c for c in record['classes'] if c])}")
            print()

    # ------------------------------------------------------------------
    # 主循环与清理
    # ------------------------------------------------------------------
    def run(self):
        print("🎬 程序已启动，在命令行中输入命令")
        try:
            while rclpy.ok() and self.running:
                rclpy.spin_once(self, timeout_sec=0.01)

                self._manage_display()

                if not self.display_proc.is_alive():
                    print("⚠️ 显示进程已意外退出，程序将终止")
                    self.running = False
                    break

                if self.exit_event.is_set():
                    print("📺 显示窗口已关闭，程序将终止")
                    self.running = False
                    break

                # 键盘轮询超时必须保持很短,否则会累积显示延迟
                key = self.keyboard_handler.get_key(0.01)
                if key:
                    self.process_command(key)

        except KeyboardInterrupt:
            print("\n\n⏹️ 程序被用户中断")
        except Exception as e:
            print(f"\n❌ 程序出错: {e}")
        finally:
            self.shutdown()

    def shutdown(self):
        print("\n🧹 正在清理资源...")
        self.exit_event.set()
        if self.display_proc.is_alive():
            self.display_proc.join(timeout=3)
        self.destroy_node()
        self.keyboard_handler.restore()
        print("👋 程序已退出")


def main():
    config = DetectorConfig.from_yaml("config/default_config.yaml")

    rclpy.init()
    try:
        detector = EnhancedDetectorNode(config)
        detector.run()
    except Exception as e:
        print(f"❌ 程序启动失败: {e}")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
