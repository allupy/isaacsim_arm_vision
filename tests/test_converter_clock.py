"""ROS integration test on domain 79, no simulation or actuator connected."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sim_scene'))
import time
import threading
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from feedback_converter import TrajectoryConverter

rclpy.init()
converter = TrajectoryConverter()
probe = Node('converter_clock_integration_test')
clock_pub = probe.create_publisher(Clock,'/clock',1)
traj_pub = probe.create_publisher(JointTrajectory,'/moveit_trajectory',10)
received=[]
probe.create_subscription(JointState,'/joint_command',lambda msg:received.append((time.monotonic(),msg.position[0])),10)
executor = MultiThreadedExecutor(num_threads=2)
executor.add_node(converter)
executor.add_node(probe)
thread=threading.Thread(target=executor.spin,daemon=True)
thread.start()
time.sleep(.5)
def clock(value):
    msg=Clock(); msg.clock.sec=int(value); msg.clock.nanosec=int((value-int(value))*1e9)
    clock_pub.publish(msg); time.sleep(.1)
def trajectory(values):
    msg=JointTrajectory(); msg.joint_names=['probe_joint']
    for value,t in values:
        point=JointTrajectoryPoint(); point.positions=[float(value)]
        point.time_from_start.sec=int(t); point.time_from_start.nanosec=int((t-int(t))*1e9)
        msg.points.append(point)
    traj_pub.publish(msg); time.sleep(.15)
try:
    clock(10.)
    trajectory([(0.,0.),(1.,1.)])
    clock(10.5)
    assert received and abs(received[-1][1]-.5)<.02,received[-4:]
    count=len(received)
    time.sleep(.3)
    assert all(abs(v-.5)<.02 for _,v in received[count:]),'Playback advanced during paused clock'
    trajectory([(.25,0.)])
    count=len(received)
    clock(11.)
    time.sleep(.2)
    assert all(abs(v-.25)<.02 for _,v in received[count:]),'Old worker overrode hold'
    print(json.dumps(dict(success=True,halfway=.5,paused_clock_holds=True,single_point_cancels=True,
                          message_count=len(received))))
finally:
    executor.shutdown()
    thread.join(timeout=2.)
    converter.destroy_node(); probe.destroy_node(); rclpy.shutdown()
