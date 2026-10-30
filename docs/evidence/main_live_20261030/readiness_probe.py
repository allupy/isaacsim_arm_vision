import json
import time
from pathlib import Path
import numpy as np
import rclpy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String
from moveit_msgs.srv import GetCartesianPath

rclpy.init()
node = rclpy.create_node('main_validation_readiness')
seen = {}
subs = []
def record(topic, msg):
    item = dict(received=time.time())
    if hasattr(msg, 'header'):
        item['age_seconds'] = time.time() - msg.header.stamp.sec - msg.header.stamp.nanosec / 1e9
    if isinstance(msg, Image):
        item.update(width=msg.width, height=msg.height, encoding=msg.encoding)
        if msg.encoding == '32FC1':
            depth = np.frombuffer(msg.data, dtype=np.float32)
            item['finite_positive_fraction'] = float(np.mean(np.isfinite(depth) & (depth > 0)))
    if isinstance(msg, JointState):
        item['joint_count'] = len(msg.position)
    seen[topic] = item
topics = {'/rgb': Image, '/depth': Image, '/observer_rgb': Image, '/observer_depth': Image,
          '/feedback/camera_calibration': String, '/feedback/joint_states': JointState}
for topic, kind in topics.items():
    subs.append(node.create_subscription(kind, topic, lambda msg, t=topic: record(t, msg), 3))
pub = node.create_publisher(String, '/feedback/camera_request', 1)
client = node.create_client(GetCartesianPath, '/compute_cartesian_path')
start = time.monotonic()
request = 0
while time.monotonic() - start < 45:
    if time.monotonic() > request:
        pub.publish(String(data='startup'))
        request = time.monotonic() + 2
    rclpy.spin_once(node, timeout_sec=.1)
    if len(seen) == len(topics) and client.service_is_ready():
        break
result = dict(topics=seen, moveit_cartesian_ready=client.service_is_ready(), missing=sorted(set(topics)-set(seen)))
Path('/experiment/readiness.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
node.destroy_node()
rclpy.shutdown()
raise SystemExit(0 if not result['missing'] and result['moveit_cartesian_ready'] else 1)
