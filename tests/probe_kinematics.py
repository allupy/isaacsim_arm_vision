"""Read-only check of USD physics poses versus MoveIt FK and measured joints."""
import json
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from moveit_msgs.srv import GetPositionFK

rclpy.init()
node = Node('feedback_kinematic_probe')
latest = {}
node.create_subscription(JointState, '/joint_states', lambda m: latest.update(joints=m), 3)
node.create_subscription(String, '/evaluation/ground_truth', lambda m: latest.update(truth=json.loads(m.data)), 3)
client = node.create_client(GetPositionFK, '/compute_fk')
deadline = time.monotonic()+25
while time.monotonic() < deadline and len(latest) < 2:
    rclpy.spin_once(node, timeout_sec=.1)
if len(latest) == 2 and client.wait_for_service(timeout_sec=3.):
    req = GetPositionFK.Request()
    req.header.frame_id = 'world'
    req.fk_link_names = ['wrist_3_link','tool0']
    js = latest['joints']
    req.robot_state.joint_state.name = list(js.name[:6])
    req.robot_state.joint_state.position = list(js.position[:6])
    future = client.call_async(req)
    rclpy.spin_until_future_complete(node, future, timeout_sec=5.)
    print('JOINTS', json.dumps(dict(zip(js.name,js.position))))
    print('USD_WRIST', json.dumps(latest['truth']['wrist']))
    if future.done() and future.result():
        print('FK_ERROR_CODE',future.result().error_code.val)
        for name, stamped in zip(future.result().fk_link_names, future.result().pose_stamped):
            p,q=stamped.pose.position,stamped.pose.orientation
            print('FK',name,[p.x,p.y,p.z],[q.x,q.y,q.z,q.w])
else:
    print('PROBE_NO_DATA',list(latest))
node.destroy_node()
rclpy.shutdown()
