#!/usr/bin/env python3
"""无头运行 workbench 场景(以引用方式组合——本机验证过可渲染):

- 空舞台 + AddReference(workbench.usd) + TestCam 式自建相机渲染已被验证可行
- /rgb /depth /observer_rgb 由本进程直接用 rclpy 发布(annotator 读数),
  绕开 ROS2CameraHelper(在本环境不稳定)
- /joint_states 与 /joint_command 由场景内 ROS_JointStates 图处理
"""
import os
import time

from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "renderer": "RayTracedLighting"})

import carb  # noqa: E402
import omni.usd  # noqa: E402
import omni.kit.app  # noqa: E402

for key in ("/physics/enforceRealtime", "/app/physics/enforceRealtime"):
    try:
        carb.settings.get_settings().set(key, True)
    except Exception:
        pass

omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(
    "isaacsim.ros2.bridge", True)

ctx = omni.usd.get_context()
ctx.open_stage("/ws/workbench.usd", None)
stage = ctx.get_stage()
print("=== STAGE OPENED (direct) ===", flush=True)

from pxr import UsdPhysics  # noqa: E402
if not any(p.GetTypeName() == "PhysicsScene" for p in stage.Traverse()):
    phys_scene = UsdPhysics.Scene.Define(stage, "/PhysicsScene")
    phys_scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    phys_scene.CreateGravityMagnitudeAttr().Set(9.81)
    print("PhysicsScene added", flush=True)

# ------------------------------------------------------------------
# 相机:rep 渲染产品 + annotator + rclpy 发布
# ------------------------------------------------------------------
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402

RP_WRIST = rep.create.render_product("/Root/ur5/wrist_3_link/WristCamera/Cam", (1280, 720))
RP_OBS = rep.create.render_product("/Root/ObserverCamera/Cam", (640, 360))
ANN_RGB = rep.AnnotatorRegistry.get_annotator("rgb")
ANN_DEPTH = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
ANN_OBS = rep.AnnotatorRegistry.get_annotator("rgb")
ANN_RGB.attach(RP_WRIST)
ANN_DEPTH.attach(RP_WRIST)
ANN_OBS.attach(RP_OBS)
print("annotators attached", flush=True)

import rclpy  # noqa: E402
from rclpy.node import Node  # noqa: E402
from sensor_msgs.msg import Image  # noqa: E402

try:
    rclpy.init()
except Exception:
    pass  # bridge 可能已初始化
ros_node = Node("sim_camera_publisher")
pub_rgb = ros_node.create_publisher(Image, "/rgb", 10)
pub_depth = ros_node.create_publisher(Image, "/depth", 10)
pub_obs = ros_node.create_publisher(Image, "/observer_rgb", 10)
print("rclpy publishers ready", flush=True)


def to_img_msg(data, encoding):
    msg = Image()
    msg.header.stamp = ros_node.get_clock().now().to_msg()
    msg.height, msg.width = data.shape[0], data.shape[1]
    msg.encoding = encoding
    msg.step = data.shape[1] * data.shape[2] if data.ndim == 3 else data.shape[1]
    msg.data = data.tobytes()
    return msg


def publish_cameras():
    """读取 annotator 并发布(数据未就绪/为空则跳过)"""
    rgb = np.asarray(ANN_RGB.get_data())
    if rgb.ndim == 3 and rgb.size:
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        pub_rgb.publish(to_img_msg(rgb[:, :, ::-1].copy(), "bgr8"))
    d = np.asarray(ANN_DEPTH.get_data())
    if d.ndim == 2 and d.size:
        pub_depth.publish(to_img_msg(np.ascontiguousarray(d, dtype=np.float32), "32FC1"))
    obs = np.asarray(ANN_OBS.get_data())
    if obs.ndim == 3 and obs.size:
        if obs.shape[-1] == 4:
            obs = obs[..., :3]
        pub_obs.publish(to_img_msg(obs[:, :, ::-1].copy(), "bgr8"))


# ------------------------------------------------------------------
# 播放循环
# ------------------------------------------------------------------
DURATION = float(os.environ.get("SIM_DURATION", "180"))

from omni.timeline import get_timeline_interface  # noqa: E402
timeline = get_timeline_interface()
timeline.play()
print("=== PLAYING ===", flush=True)

CAM_EVERY = int(os.environ.get("CAM_EVERY", "4"))  # 每 N 帧发布一次相机
t0 = time.time()
n = 0
cam_frames = 0
while time.time() - t0 < DURATION:
    app.update()
    n += 1
    if n % CAM_EVERY == 0:
        publish_cameras()
        cam_frames += 1
    if n % 600 == 0:
        print(f"sim running {time.time() - t0:.0f}s, {n} frames, cam_pub={cam_frames}", flush=True)
    if n == 600 and os.environ.get("DEBUG_RENDER", "0") == "1":
        try:
            import cv2
            rgb = np.asarray(ANN_OBS.get_data())
            if rgb.ndim == 3 and rgb.size:
                cv2.imwrite("/ws/_dbg_live_obs.png", rgb[..., :3][:, :, ::-1].copy())
                print(f"DEBUG_RENDER live obs mean={rgb.mean():.1f}", flush=True)
            wr = np.asarray(ANN_RGB.get_data())
            if wr.ndim == 3 and wr.size:
                cv2.imwrite("/ws/_dbg_live_wrist.png", wr[..., :3][:, :, ::-1].copy())
                print(f"DEBUG_RENDER live wrist mean={wr.mean():.1f}", flush=True)
        except Exception as e:
            print("DEBUG_RENDER error:", e, flush=True)

timeline.stop()
print(f"=== SIM DONE (cam_pub={cam_frames}) ===", flush=True)
app.close()
