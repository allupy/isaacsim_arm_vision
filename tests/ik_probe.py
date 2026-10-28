import json,time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from moveit_msgs.srv import GetPositionIK
rclpy.init()
node=Node('ik_diagnostic_probe')
last=[]
node.create_subscription(JointState,'/joint_states',lambda m:last.append(m),1)
end=time.monotonic()+10
while not last and time.monotonic()<end:
    rclpy.spin_once(node,timeout_sec=.1)
client=node.create_client(GetPositionIK,'/compute_ik')
client.wait_for_service(timeout_sec=3.)
for link in ['wrist_3_link','tool0','']:
    request=GetPositionIK.Request();ik=request.ik_request
    ik.group_name='ur5_arm';ik.ik_link_name=link
    ik.pose_stamped.header.frame_id='world'
    ik.pose_stamped.pose.position.x=.45;ik.pose_stamped.pose.position.y=-.12;ik.pose_stamped.pose.position.z=.5
    ik.pose_stamped.pose.orientation.x=1.
    ik.pose_stamped.pose.orientation.w=0.
    ik.robot_state.joint_state=last[-1]
    ik.avoid_collisions=True;ik.timeout.sec=2
    future=client.call_async(request)
    rclpy.spin_until_future_complete(node,future,timeout_sec=5.)
    response=future.result() if future.done() else None
    print(json.dumps(dict(link=link,code=response.error_code.val if response else None,
                         solution=list(response.solution.joint_state.position) if response else None)),flush=True)
node.destroy_node();rclpy.shutdown()
