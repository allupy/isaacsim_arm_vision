#!/usr/bin/env python3
"""
增强版目标检测器 - 显示独立进程版本（修复3秒显示问题）
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import threading
import sys
import numpy as np 
import time
import tf2_ros
import math
import cv2
import multiprocessing as mp  # 导入多进程模块

from modules.detection_manager import DetectionManager
from modules.keyboard_handler import KeyboardHandler
from modules.io_manager import IOManager
from modules.class_manager import ClassManager
from modules.utils import quaternion_to_euler,build_transform_from_matrix,build_transform,quaternion_to_matrix
from modules.arm_manager import ArmControllerWithGripper
from config.config import DetectorConfig

# ----------------------
# 独立的显示进程函数
# ----------------------
def display_process(image_queue, exit_event, window_name="Detection"):
    """
    独立的显示进程
    :param image_queue: 图像队列，用于接收主进程的图像数据
    :param exit_event: 退出事件，用于通知进程退出
    :param window_name: 显示窗口名称
    """
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    
    print(f"📺 显示进程已启动 (PID: {mp.current_process().pid})")
    
    while not exit_event.is_set():
        try:
            # 非阻塞获取队列中的图像
            if not image_queue.empty():
                img = image_queue.get_nowait()
                cv2.imshow(window_name, img)
            
            # 处理键盘事件（必须在显示进程中处理）
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC键退出
                exit_event.set()
                
        except Exception as e:
            print(f"⚠️ 显示进程出错: {e}")
            continue
    
    # 清理资源
    cv2.destroyAllWindows()
    print("📺 显示进程已退出")

class EnhancedDetectorNode(Node):
    """ROS2检测节点"""
    
    def __init__(self, config: DetectorConfig):
        super().__init__('enhanced_detector')
        
        # 初始化各个模块
        self.config = config
        self.bridge = CvBridge()

        # TF相关,机器人位置姿态相关
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)      

        # 坐标系配置
        self.ee_frame = 'wrist_3_link'  # 末端执行器坐标系
        self.base_frame = 'base_link'   # 机器人基座标系
        self.R_cam2gripper, self.t_cam2gripper = config.get_cam2gripper_pose() #相机相对于末端执行器的位置
        
        self.camera_matrix = np.array(
                [[634.165821427196, 0.0, 636.8610491500791], 
                 [0.0, 634.0985503275863, 357.7665813647669], 
                 [0.0, 0.0, 1.0]
        ])   
        self.dist_coeffs = np.array([0.0028141594696008986, -0.006156795039937514, -0.0006492150364271804, -0.0012946439050663806, 0.003993861075015315])

        # 存储同步的图像对
        self.current_rgb = None
        self.current_depth = None
        self.current_robot_pose = None 

        # ----------------------
        # 显示状态管理（修复3秒显示核心）
        # ----------------------
        self.show_result = False          # 是否显示检测结果帧
        self.result_img = None            # 检测结果帧缓存
        self.result_show_start = 0.0      # 结果帧开始显示的时间
        self.result_show_duration = 3.0   # 结果帧显示时长（秒）
        self.last_display_img = None      # 上一次发送的显示图像（避免重复发送）

        # ----------------------
        # 多进程显示相关初始化
        # ----------------------
        self.mp_context = mp.get_context('spawn')  # 使用spawn方式创建进程（ROS2兼容）
        self.image_queue = self.mp_context.Queue(maxsize=5)  # 图像队列（最大缓存5帧）
        self.exit_event = self.mp_context.Event()  # 退出事件
        
        # 启动显示进程
        self.display_proc = self.mp_context.Process(
            target=display_process,
            args=(self.image_queue, self.exit_event, "Detection")
        )
        self.display_proc.start()

        # 模块初始化
        self.io_manager = IOManager(config.save_dir)
        self.keyboard_handler = KeyboardHandler()
        self.arm_controller = ArmControllerWithGripper(self)
        
        # 类别管理器
        self.class_manager = ClassManager(config.default_classes)
        self.class_manager.on_classes_changed = self._on_classes_changed
        
        # 检测管理器
        self.detection_manager = DetectionManager(config)
        
        # 初始化订阅器
        self._setup_subscribers()

        # 状态
        self.running = True
        
        # 打印欢迎信息
        self._print_welcome()
    
    def _setup_subscribers(self):
        """设置订阅器（无时间同步版本）"""
        # 创建独立订阅
        self.rgb_sub = self.create_subscription(
            Image, 
            '/rgb', 
            self.rgb_callback, 
            10
        )
        
        self.depth_sub = self.create_subscription(
            Image, 
            '/depth', 
            self.depth_callback, 
            10
        )
        
        self.get_logger().info("已启用RGB-D订阅（无时间同步）")
        
    def rgb_callback(self, msg):
        """RGB图像回调（修改：不再直接发送到显示队列）"""
        try:
            self.current_rgb = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"RGB回调失败: {e}")
            
    def depth_callback(self, msg):
        """深度图像回调"""
        try:
            self.current_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().error(f"深度回调失败: {e}")

    def send_to_display(self, img):
        """
        发送图像到显示进程（优化：避免重复发送相同图像）
        :param img: OpenCV图像（BGR格式）
        """
        try:
            # 避免重复发送相同图像（减少队列压力）
            if self.last_display_img is None or not np.array_equal(self.last_display_img, img):
                if not self.image_queue.full():
                    # 复制图像避免内存共享问题
                    self.image_queue.put_nowait(img.copy())
                    self.last_display_img = img.copy()
        except Exception as e:
            self.get_logger().warn(f"发送图像到显示进程失败: {e}")

    def get_current_frames(self):
        """获取当前帧的RGB、深度图像和机器人位姿"""
        rgb = self.current_rgb.copy() if self.current_rgb is not None else None
        depth = self.current_depth.copy() if self.current_depth is not None else None
        success, robot_pose = self.get_robot_pose()
        if success:
            self.current_robot_pose = robot_pose.copy()
            pose = self.current_robot_pose if self.current_robot_pose is not None else None
        
        return rgb, depth, pose
            
    def _print_welcome(self):
        """打印欢迎信息"""
        print("\n" + "="*60)
        print("🎯 增强版目标检测器 - 显示独立进程版本（修复3秒显示）")
        print("="*60)
        print(f"\n当前检测类别: {self.class_manager.format_classes_display()}")
        print(f"保存目录: {self.io_manager.save_dir.absolute()}")
        print(f"显示进程PID: {self.display_proc.pid}")
        self._print_help()

    def _print_help(self):
        """打印帮助信息"""
        print("\n📋 快捷键:")
        print("  [d] 实时检测当前帧")
        print("  [s] 拍照并保存检测结果")
        print("  [c] 修改检测类别")
        print("  [r] 重置为默认类别/机械臂归位")
        print("  [l] 显示历史记录")
        print("  [h] 显示此帮助")
        print("  [q] 退出程序")
        print("  [ESC] 关闭显示窗口")
        print("\n💡 提示: 在命令行中输入命令")
        print("="*60)

    def _on_classes_changed(self, classes: list):
        """类别变更回调"""
        self.detection_manager.set_classes(classes)
        print(f"\n✅ 检测类别已更新: {self.class_manager.format_classes_display()}")
  
    def get_robot_pose(self):
        """获取机器人末端执行器位姿"""
        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.ee_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
            
            t = transform.transform.translation
            r = transform.transform.rotation
            
            # 获取旋转矩阵
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

    def process_command(self, command: str):
        """处理命令行命令"""
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
        """命令：实时检测（修复3秒显示逻辑）"""
        rgb, depth, robot_pose = self.get_current_frames()
        if rgb is None:
            print("❌ 错误: 没有可用的图像帧")
            return
        
        print("\n🎯 运行实时检测...")
        result_img, detections = self.detection_manager.detect(rgb)

        if result_img is not None:
            # 初始化检测结果显示状态
            self.result_img = result_img.copy()
            self.show_result = True
            self.result_show_start = time.time()  # 记录开始显示时间
            # 立即发送检测结果到显示进程
            self.send_to_display(self.result_img)
            print(f"✅ 检测完成，找到 {len(detections)} 个目标（将显示{self.result_show_duration}秒）")
        
        # 选择置信度最高的目标
        if len(detections) > 0:
            sorted_detections = sorted(
                detections, 
                key=lambda d: d.confidence, 
                reverse=True
            )
            
            best_detection = sorted_detections[0]
            mask = best_detection.mask

            if mask is not None and mask.any():
                try:
                    x, y, z = self.pose_estimation(rgb, depth, robot_pose, mask)
                    print(f"物体位置在({x:.3f},{y:.3f},{z:.3f})")

                    # 机械臂操作在主进程执行（阻塞但不影响显示）
                    if self.arm_controller.execute_pick_place(
                        [x, y, 0.0],
                        place_position=[0.0,-0.5,0.10],
                    ):
                        print("✅ 抓取完成")
                    else:
                        print("❌ 抓取失败")
                except ZeroDivisionError as e:
                    print(f"计算姿态时出错: {e}，掩码可能有问题")
                except Exception as e:
                    print(f"姿态估计失败: {e}")
            else:
                print(f"警告: 检测到{best_detection.class_name}但没有掩码，无法进行姿态估计")

    def pose_estimation(self, rgb, depth, ee_in_base, mask):
        """位置姿态估计函数"""
        camera_matrix = self.camera_matrix
        dist_coeffs = self.dist_coeffs
        
        # 1. 计算质心像素坐标
        M = cv2.moments(mask)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        
        # 2. 从深度图获取物体距离
        depth_value = depth[cy, cx]
        Z = depth_value
        
        # 3. 像素坐标 -> 相机坐标系3D坐标
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx_cam = camera_matrix[0, 2]
        cy_cam = camera_matrix[1, 2]
        
        X_cam = (cx - cx_cam) * Z / fx
        Y_cam = (cy - cy_cam) * Z / fy
        Z_cam = Z
        
        point_camera = np.array([X_cam, Y_cam, Z_cam, 1.0]).reshape(4, 1)

        # 4. 转换到基坐标系
        R_ee2base = ee_in_base['rotation_matrix']
        t_ee2base = np.array(ee_in_base['position']).flatten()
        T_ee2base = build_transform_from_matrix(t_ee2base, R_ee2base)

        R_cam2ee = self.R_cam2gripper
        t_cam2ee = self.t_cam2gripper.flatten()
        T_cam2ee = build_transform_from_matrix(t_cam2ee, R_cam2ee)

        T_cam2base = T_ee2base @ T_cam2ee
        point_base_homogeneous = T_cam2base @ point_camera
        point_base = point_base_homogeneous[:3].flatten()

        x_world = point_base[0]
        y_world = point_base[1]
        z_world = point_base[2]
               
        return (x_world, y_world, z_world)
        
    def _command_save(self):
        """命令：拍照并保存"""
        rgb = self.current_rgb
        if rgb is None:
            print("❌ 错误: 没有可用的图像帧")
            return
        
        print("\n📸 拍照并保存检测结果...")
        
        result_img, detections = self.detection_manager.detect(rgb)
        
        if result_img is not None:
            # 临时显示保存结果3秒
            self.result_img = result_img.copy()
            self.show_result = True
            self.result_show_start = time.time()
            self.send_to_display(self.result_img)
            
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
        """命令：修改类别"""
        print("\n" + "="*60)
        print("📝 修改检测类别")
        print("="*60)
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
        """命令：显示历史"""
        print("\n" + "="*60)
        print("📊 检测历史")
        print("="*60)
        
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
    
    def _manage_display(self):
        """统一管理显示内容（核心修复：保证3秒显示）"""
        current_time = time.time()
        
        # 1. 如果需要显示检测结果且在时长内
        if self.show_result and self.result_img is not None:
            elapsed_time = current_time - self.result_show_start
            
            # 还在显示时长内：持续发送检测结果帧
            if elapsed_time < self.result_show_duration:
                self.send_to_display(self.result_img)
            # 超出时长：切换回实时帧
            else:
                self.show_result = False
                self.result_img = None
                self.last_display_img = None  # 重置缓存
        # 2. 显示实时帧
        elif self.current_rgb is not None:
            self.send_to_display(self.current_rgb)

    def run(self):
        """主循环（新增显示管理调用）"""
        print("🎬 程序已启动，在命令行中输入命令")

        try:
            while rclpy.ok() and self.running:
                # 处理ROS消息
                rclpy.spin_once(self, timeout_sec=0.01)

                # ----------------------
                # 核心修复：调用显示管理逻辑
                # ----------------------
                self._manage_display()

                # 检查显示进程是否还在运行
                if not self.display_proc.is_alive():
                    print("⚠️ 显示进程已意外退出，程序将终止")
                    self.running = False
                    break

                # 检查退出事件（显示进程触发）
                if self.exit_event.is_set():
                    print("📺 显示窗口已关闭，程序将终止")
                    self.running = False
                    break

                # 检查键盘输入
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
        """清理资源"""
        print("\n🧹 正在清理资源...")
        
        # 通知显示进程退出
        self.exit_event.set()
        
        # 等待显示进程结束
        if self.display_proc.is_alive():
            self.display_proc.join(timeout=3)
        
        # 清理ROS节点
        self.destroy_node()
        
        # 清理其他资源
        self.keyboard_handler.restore()
        
        print("👋 程序已退出")

# main.py 部分
#!/usr/bin/env python3
"""
主程序入口
"""
import rclpy
from config.config import DetectorConfig

def main():
    # 加载配置
    config = DetectorConfig.from_yaml("config/default_config.yaml")
    
    # 初始化ROS2
    rclpy.init()
    
    try:
        # 创建节点
        detector = EnhancedDetectorNode(config)     
        # 运行
        detector.run()
        
    except Exception as e:
        print(f"❌ 程序启动失败: {e}")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()