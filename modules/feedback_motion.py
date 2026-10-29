"""Measured completion for the existing MoveIt/IK interfaces (ROS2 adapter)."""
import math
import time
import json
import numpy as np
import rclpy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.srv import GetPositionIK, GetCartesianPath
from moveit_msgs.msg import PositionIKRequest, RobotState
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
from rosgraph_msgs.msg import Clock
from std_msgs.msg import String
from modules.moveit_controller import MoveItArmController
from modules.joint_geometry import nearest_equivalent
from modules.trajectory_playback import retime_cartesian


class FeedbackArmController(MoveItArmController):
    def __init__(self, node, params=None):
        self.measured_at = 0.0
        self.measured_sequence = 0
        self.last_motion = {}
        self.time_scale = float((params or {}).get('trajectory_time_scale', 1.))
        if not math.isfinite(self.time_scale) or self.time_scale < 1.:
            raise ValueError('trajectory_time_scale must be finite and >= 1; tune bounded motion profiles for speed')
        self.optimized = (params or {}).get('optimized', True)
        self.motion_phase = ''
        self.sim_time = None
        self.clock_sub = node.create_subscription(Clock, '/clock', self._on_sim_clock, 1)
        readiness = node.create_client(GetPositionIK, '/compute_ik')
        ready = readiness.wait_for_service(timeout_sec=15.)
        node.destroy_client(readiness)
        if not ready:
            raise TimeoutError('MoveIt /compute_ik service unavailable after 15s')
        super().__init__(node, params=params)
        self.physics_trajectory_pub = node.create_publisher(String,'/feedback/timed_trajectory',1)
        node.destroy_subscription(self.joint_state_sub)
        self.joint_state_sub = node.create_subscription(JointState, '/feedback/joint_states', self.joint_state_callback, 1)
        self.cartesian_client = node.create_client(GetCartesianPath, '/compute_cartesian_path')
        if self.optimized:
            self.plan_params.max_velocity_scaling_factor = .7
            self.plan_params.max_acceleration_scaling_factor = .65

    def _on_sim_clock(self, msg):
        self.sim_time = msg.clock.sec + msg.clock.nanosec*1e-9

    def pump(self):
        rclpy.spin_once(self.node, timeout_sec=0.02)

    def publish_trajectory(self,msg):
        if self.optimized:
            deadline = time.monotonic()+3.
            while self.physics_trajectory_pub.get_subscription_count() == 0 and time.monotonic() < deadline:
                self.pump()
            if self.physics_trajectory_pub.get_subscription_count() == 0:
                raise TimeoutError('physics_trajectory_receiver_missing')
        self.trajectory_pub.publish(msg)
        payload=dict(names=list(msg.joint_names),
                     times=[p.time_from_start.sec+p.time_from_start.nanosec*1e-9 for p in msg.points],
                     positions=[list(p.positions) for p in msg.points])
        if all(len(p.velocities)==len(msg.joint_names) for p in msg.points):
            payload['velocities']=[list(p.velocities) for p in msg.points]
        self.physics_trajectory_pub.publish(String(data=json.dumps(payload)))

    def joint_state_callback(self, msg):
        if not all(name in msg.name for name in self.joint_names[:6]):
            return
        if len(msg.name) != len(msg.position) or not np.isfinite(msg.position).all():
            return
        super().joint_state_callback(msg)
        self.measured_at = time.monotonic()
        self.measured_sequence += 1

    def call(self, client, request, timeout=6.):
        if not client.wait_for_service(timeout_sec=2.):
            return None
        future = client.call_async(request)
        end = time.monotonic()+timeout
        while rclpy.ok() and not future.done() and time.monotonic() < end:
            self.pump()
        if future.done():
            return future.result()
        future.cancel()
        return None

    def compute_ik_single(self, position, orientation, seed):
        request = GetPositionIK.Request()
        ik = request.ik_request
        ik.group_name = 'ur5_arm'
        ik.ik_link_name = 'wrist_3_link'
        ik.pose_stamped.header.frame_id = 'world'
        ik.pose_stamped.pose.position.x, ik.pose_stamped.pose.position.y, ik.pose_stamped.pose.position.z = map(float, position)
        ik.pose_stamped.pose.orientation.x, ik.pose_stamped.pose.orientation.y, ik.pose_stamped.pose.orientation.z, ik.pose_stamped.pose.orientation.w = map(float, orientation)
        ik.robot_state.joint_state.name = self.joint_names[:6]
        ik.robot_state.joint_state.position = list(seed)
        ik.avoid_collisions = True
        ik.timeout.sec = 2
        response = self.call(self.ik_client, request)
        if response is None or response.error_code.val != 1:
            self.last_ik_error = 'timeout' if response is None else response.error_code.val
            self.node.get_logger().warn(f'IK failed for wrist_3_link: {self.last_ik_error}, seed={seed}')
            return None
        js = response.solution.joint_state
        if not all(n in js.name for n in self.joint_names[:6]):
            return None
        return [js.position[js.name.index(n)] for n in self.joint_names[:6]]

    def _plan_to_pose(self, position, orientation):
        """Seed IK with measured joints, then plan to that explicit joint goal.

        A fresh pose-only MoveItPy goal can select a remote IK branch even when
        the requested TCP pose is already the current pose.
        """
        started = time.monotonic()
        from moveit.core.robot_state import RobotState as MoveItRobotState
        seed = self.get_current_joint_positions()
        solution = self.compute_ik_single(position,orientation,seed)
        if solution is None:
            return None
        limits = [(-2*math.pi,2*math.pi),(-2*math.pi,2*math.pi),(-math.pi,math.pi),
                  (-2*math.pi,2*math.pi),(-2*math.pi,2*math.pi),(-2*math.pi,2*math.pi)]
        normalized = [nearest_equivalent(q,ref,*bounds) for q,ref,bounds in zip(solution,seed,limits)]
        state = MoveItRobotState(self.moveit.get_robot_model())
        state.joint_positions = dict(zip(self.joint_names[:6],normalized))
        state.update()
        self.planning_component.set_start_state_to_current_state()
        self.planning_component.set_goal_state(robot_state=state)
        result = self.planning_component.plan(parameters=self.plan_params)
        self.last_plan_wall_s = time.monotonic()-started
        if not result:
            return None
        return result.trajectory.get_robot_trajectory_msg().joint_trajectory

    def wait_measured(self, names, positions, timeout, command_start, tolerance=0.04, not_before=None, sim_not_before=None):
        deadline = time.monotonic()+timeout
        stable_since = None
        evidence = []
        last_sequence = self.measured_sequence
        while rclpy.ok() and time.monotonic() < deadline:
            self.pump()
            if self.measured_sequence == last_sequence or self.measured_at < command_start:
                continue
            if not_before is not None and time.monotonic() < not_before:
                continue
            if sim_not_before is not None and (self.sim_time is None or self.sim_time < sim_not_before):
                continue
            last_sequence = self.measured_sequence
            js = self.latest_joint_state
            if js is None or not all(n in js.name for n in names):
                continue
            actual = [js.position[js.name.index(n)] for n in names]
            raw_error = max(abs(a-b) for a, b in zip(actual, positions))
            error = max(abs(math.atan2(math.sin(a-b),math.cos(a-b))) for a,b in zip(actual,positions))
            evidence.append(error)
            if error < tolerance:
                stable_since = stable_since or time.monotonic()
                if time.monotonic()-stable_since > (.15 if self.optimized else .4):
                    self.last_motion = dict(success=True, error=error, raw_error=raw_error, samples=len(evidence))
                    return True
            else:
                stable_since = None
        self.last_motion = dict(success=False, reason='joint_convergence_timeout',
                                error=evidence[-1] if evidence else None, samples=len(evidence),
                                measured=self.get_current_joint_positions(), target=list(positions))
        return False

    def _send_trajectory(self, traj, gripper_position=None, extra_wait=.5):
        if not traj.points or not traj.joint_names:
            return False
        if gripper_position is None:
            gripper_position = self.current_gripper
        msg = JointTrajectory()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.joint_names = list(traj.joint_names)+['finger_joint']
        for pt in traj.points:
            out = JointTrajectoryPoint()
            out.positions = list(pt.positions)+[float(gripper_position)]
            if self.optimized and len(pt.velocities)==len(traj.joint_names):
                out.velocities = [v/self.time_scale for v in pt.velocities]+[0.]
            scaled = (pt.time_from_start.sec+pt.time_from_start.nanosec/1e9)*self.time_scale
            out.time_from_start.sec = int(scaled)
            out.time_from_start.nanosec = int((scaled-int(scaled))*1e9)
            msg.points.append(out)
        duration = msg.points[-1].time_from_start.sec+msg.points[-1].time_from_start.nanosec/1e9
        for _ in range(5):
            self.pump()
        if self.sim_time is None:
            self.last_motion = dict(success=False,reason='simulation_clock_missing')
            return False
        sim_start = self.sim_time
        if hasattr(self.node,'output'):
            import json
            trace = dict(names=msg.joint_names,sim_start=sim_start,
                         points=[dict(t=p.time_from_start.sec+p.time_from_start.nanosec/1e9,q=list(p.positions)) for p in msg.points])
            with (self.node.output/'trajectories.jsonl').open('a') as stream:
                stream.write(json.dumps(trace)+'\n')
        command_start = time.monotonic()
        self.publish_trajectory(msg)
        self.current_gripper = gripper_position
        success = self.wait_measured(traj.joint_names, traj.points[-1].positions,
                                  max(60., duration*15+20), command_start, sim_not_before=sim_start+duration)
        self.last_motion.update(planning_wall_s=getattr(self,'last_plan_wall_s',0.),
            execution_wall_s=time.monotonic()-command_start, trajectory_sim_s=duration,
            sim_elapsed_s=None if self.sim_time is None else self.sim_time-sim_start)
        return success

    def move_to_position_direct(self, position, gripper_position=None, duration=1.,
                                orientation_type='down', orientation=None):
        # No single-IK fallback: a failed path is not a valid path.
        trajectory = self._plan_to_pose(position, self._resolve_orientation(orientation_type, orientation))
        if trajectory is None:
            self.last_motion = dict(success=False, reason='planning_failed')
            return False
        success = self._send_trajectory(trajectory, gripper_position)
        if success:
            self.current_position = list(position)
        return success

    def cartesian_to(self, position, orientation):
        started = time.monotonic()
        request = GetCartesianPath.Request()
        request.header.frame_id = 'world'
        request.group_name = 'ur5_arm'
        request.link_name = 'wrist_3_link'
        request.start_state.joint_state.name = self.joint_names[:6]
        request.start_state.joint_state.position = self.get_current_joint_positions()
        request.max_step = .01
        request.jump_threshold = 1.5
        request.avoid_collisions = True
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = map(float, position)
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = map(float, orientation)
        request.waypoints = [pose]
        response = self.call(self.cartesian_client, request, timeout=10.)
        if response is None or response.error_code.val != 1 or response.fraction < .999:
            self.last_motion = dict(success=False, reason='cartesian_path_incomplete',
                                     fraction=None if response is None else response.fraction)
            return False
        trajectory = response.solution.joint_trajectory
        if not trajectory.points:
            return False
        self.last_plan_wall_s = time.monotonic()-started
        if self.optimized:
            contact = self.motion_phase in ['DESCEND','PLACE']
            velocity=.30 if contact else .55
            acceleration=.6 if contact else 1.
            native_times=[p.time_from_start.sec+p.time_from_start.nanosec*1e-9 for p in trajectory.points]
            # The installed Cartesian service already runs MoveIt's TOTG. Preserve
            # that profile instead of overwriting it with waypoint-by-waypoint timing.
            if (len(native_times)>1 and np.all(np.diff(native_times)>0) and
                    all(len(p.velocities)==6 and len(p.accelerations)==6 for p in trajectory.points)):
                vmax=max(abs(v) for p in trajectory.points for v in p.velocities)
                amax=max(abs(a) for p in trajectory.points for a in p.accelerations)
                factor=max(1.,vmax/velocity,math.sqrt(amax/acceleration))
                for point,t in zip(trajectory.points,native_times):
                    t *= factor
                    point.time_from_start.sec=int(t)
                    point.time_from_start.nanosec=int((t-int(t))*1e9)
                    point.velocities=[v/factor for v in point.velocities]
                    point.accelerations=[a/(factor*factor) for a in point.accelerations]
                return self._send_trajectory(trajectory)
            times, values = retime_cartesian([self.get_current_joint_positions()]+
                [list(p.positions) for p in trajectory.points],
                velocity=velocity, acceleration=acceleration)
            trajectory.points = []
            for t, q in zip(times, values):
                point = JointTrajectoryPoint()
                point.positions = q
                point.time_from_start.sec = int(t)
                point.time_from_start.nanosec = int((t-int(t))*1e9)
                trajectory.points.append(point)
            return self._send_trajectory(trajectory)
        # Legacy timing retained for controlled comparison.
        previous = self.get_current_joint_positions()
        elapsed = 0.
        for pt in trajectory.points:
            elapsed += max(.08, max(abs(a-b) for a,b in zip(pt.positions,previous))/.30)
            pt.time_from_start.sec = int(elapsed)
            pt.time_from_start.nanosec = int((elapsed-int(elapsed))*1e9)
            previous = pt.positions
        return self._send_trajectory(trajectory)

    def set_gripper(self, closed):
        goal = .78 if closed else 0.
        # Skip an already open gripper only with a fresh measured sample.
        if self.optimized and not closed and time.monotonic()-self.measured_at < .5:
            js = self.latest_joint_state
            if js is not None and 'finger_joint' in js.name and abs(js.position[js.name.index('finger_joint')]) < .03:
                self.current_gripper = goal
                return True
        message = JointTrajectory()
        message.joint_names = self.joint_names
        point = JointTrajectoryPoint()
        point.positions = self.get_current_joint_positions()+[goal]
        point.time_from_start.sec = 1
        message.points = [point]
        if self.optimized:
            js = self.latest_joint_state
            if js is None or 'finger_joint' not in js.name or time.monotonic()-self.measured_at > .5:
                return False
            initial = JointTrajectoryPoint()
            initial.positions = self.get_current_joint_positions()+[js.position[js.name.index('finger_joint')]]
            initial.velocities = [0.]*7
            point.velocities = [0.]*7
            point.time_from_start.sec = 0
            point.time_from_start.nanosec = 600000000
            message.points = [initial,point]
        start = time.monotonic()
        self.publish_trajectory(message)
        self.current_gripper = goal
        # Closing against an object can stop before the commanded joint limit.
        if not closed:
            return self.wait_measured(['finger_joint'], [goal], 15., start, .06)
        last = None
        stable = None
        sequence = self.measured_sequence
        while time.monotonic()-start < 15.:
            self.pump()
            js = self.latest_joint_state
            if self.measured_sequence == sequence or js is None or 'finger_joint' not in js.name:
                continue
            sequence = self.measured_sequence
            value = js.position[js.name.index('finger_joint')]
            if value > .05 and last is not None and abs(value-last) < .015:
                stable = stable or time.monotonic()
                if time.monotonic()-stable > .5:
                    return True
            else:
                stable = None
            last = value
        return False

    def hold(self):
        if time.monotonic()-self.measured_at > 2.:
            return False
        msg = JointTrajectory()
        msg.joint_names = self.joint_names
        point = JointTrajectoryPoint()
        point.positions = self.get_current_joint_positions()+[self.current_gripper]
        msg.points = [point]
        self.publish_trajectory(msg)
        return True
