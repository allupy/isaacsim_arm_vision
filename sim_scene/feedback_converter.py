# ur5_moveit_config/launch/trajectory_converter.py
"""
轨迹转换器(升级版):/moveit_trajectory (JointTrajectory) → /joint_command (JointState)

原版只转发轨迹的第一个点,导致任何多点轨迹都被截断成单点命令——
这是旧方案只能"逐点发送+sleep"并伴随启停卡顿的直接原因。

升级后:
  - 单点轨迹:立即发布(与旧行为完全兼容);
  - 多点轨迹:按 time_from_start 时间戳做 60Hz 线性插值回放,
    MoveIt 规划出的时间最优轨迹得以平滑执行;
  - 新轨迹到来时自动打断当前回放。

回放线程读取 /clock 的仿真时间进行插值;无时钟时拒绝多点回放,
仿真暂停时保持当前点。单点停止/夹爪命令立即生效并取消旧回放。
"""
import threading
import time

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory
from sensor_msgs.msg import JointState
from rosgraph_msgs.msg import Clock

PLAYBACK_RATE = 60.0  # 回放频率 Hz


def _pt_time(pt) -> float:
    return pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9


class TrajectoryConverter(Node):
    def __init__(self):
        super().__init__('trajectory_converter')

        self.publisher = self.create_publisher(JointState, '/joint_command', 10)
        self.subscription = self.create_subscription(
            JointTrajectory, '/moveit_trajectory', self.trajectory_callback, 10)

        self._play_lock = threading.Lock()
        self._play_generation = 0  # 代数号:新轨迹使旧回放线程自然退出
        self._play_last_log = 0.0
        self._sim_time = None
        self.clock_sub = self.create_subscription(Clock, '/clock', self._clock_callback, 1)

        self.get_logger().info('🚀 轨迹转换器(整条轨迹插值回放版)启动完成')

    # ------------------------------------------------------------------
    def _clock_callback(self, msg):
        self._sim_time = msg.clock.sec + msg.clock.nanosec*1e-9

    def trajectory_callback(self, msg: JointTrajectory):
        if not msg.points:
            return

        if len(msg.points) == 1:
            # A stop/gripper command must also cancel an old playback worker.
            with self._play_lock:
                self._play_generation += 1
                self._publish_positions(msg.joint_names, list(msg.points[0].positions))
            return

        # 多点轨迹:起墙钟回放线程
        self._start_playback(msg)

    # ------------------------------------------------------------------
    def _start_playback(self, msg: JointTrajectory):
        if self._sim_time is None:
            self.get_logger().error('No /clock: refusing timed playback')
            return
        with self._play_lock:
            self._play_generation += 1
            generation = self._play_generation

        total = _pt_time(msg.points[-1])
        self.get_logger().info(
            f'📥 开始回放轨迹: {len(msg.points)} 点, 时长 {total:.2f}s')

        thread = threading.Thread(
            target=self._playback_worker,
            args=(msg, generation, total),
            daemon=True,
        )
        thread.start()

    def _playback_worker(self, traj: JointTrajectory, generation: int, total: float):
        """线程轮询仿真时钟;代数号变化时取消旧回放。"""
        t0 = self._sim_time
        dt = 1.0 / PLAYBACK_RATE
        pts = traj.points

        while True:
            with self._play_lock:
                if generation != self._play_generation:
                    self.get_logger().info('⏭️ 回放被新轨迹打断')
                    return

                t = self._sim_time - t0
                if t >= total:
                    self._publish_positions(traj.joint_names, list(pts[-1].positions))
                    return
                pos = self._interpolate(pts, t)
                self._publish_positions(traj.joint_names, pos)
            time.sleep(dt)

    @staticmethod
    def _interpolate(pts, t: float):
        """t 时刻的关节位置(分段线性插值)"""
        for i in range(len(pts) - 1):
            t1, t2 = _pt_time(pts[i]), _pt_time(pts[i + 1])
            if t1 <= t <= t2:
                if t2 > t1:
                    a = (t - t1) / (t2 - t1)
                    return [p1 + (p2 - p1) * a
                            for p1, p2 in zip(pts[i].positions, pts[i + 1].positions)]
                return list(pts[i + 1].positions)
        return list(pts[0].positions)

    def _publish_positions(self, names, positions):
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = names
        js.position = positions
        self.publisher.publish(js)
        # 节流日志
        now = time.monotonic()
        if now - self._play_last_log > 2.0:
            self._play_last_log = now
            self.get_logger().info(f'📤 回放中... {len(positions)} 关节')


def main():
    rclpy.init()
    node = TrajectoryConverter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
