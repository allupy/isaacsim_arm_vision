#!/usr/bin/env python3
"""话题视频录制器:ros2 图像话题 → mp4(用法: recorder.py <topic> <out.mp4> [fps])"""
import signal
import sys

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

topic = sys.argv[1]
out = sys.argv[2]
fps = float(sys.argv[3]) if len(sys.argv) > 3 else 15.0

rclpy.init()
node = Node('video_recorder')
bridge = CvBridge()
state = {'writer': None, 'frames': 0}


def cb(msg):
    img = bridge.imgmsg_to_cv2(msg, 'bgr8')
    if state['writer'] is None:
        h, w = img.shape[:2]
        state['writer'] = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'),
                                          fps, (w, h))
        print(f"REC_START {out} {w}x{h} @{fps}", flush=True)
    state['writer'].write(img)
    state['frames'] += 1


node.create_subscription(Image, topic, cb, 10)


def stop(*_):
    raise KeyboardInterrupt


signal.signal(signal.SIGINT, stop)
signal.signal(signal.SIGTERM, stop)

try:
    rclpy.spin(node)
except KeyboardInterrupt:
    pass
finally:
    if state['writer'] is not None:
        state['writer'].release()
    print(f"RECORDER_SAVED {out} frames={state['frames']}", flush=True)
    node.destroy_node()
    rclpy.shutdown()
