import os, time
from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "renderer": "RayTracedLighting"})

import carb  # noqa: E402
import omni.usd  # noqa: E402
import omni.kit.app  # noqa: E402

omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(
    "isaacsim.ros2.bridge", True)

for key in ("/physics/enforceRealtime", "/app/physics/enforceRealtime"):
    try:
        carb.settings.get_settings().set(key, True)
    except Exception:
        pass

import numpy as np  # noqa: E402
import rclpy as _rclpy  # noqa: E402
from rclpy.node import Node as _Node  # noqa: E402
from sensor_msgs.msg import Image as _Image  # noqa: E402

ctx = omni.usd.get_context()
ctx.open_stage(os.environ.get("STAGE_FILE", "/ws/workbench.usd"), None)
print("=== STAGE OPENED ===", flush=True)

from omni.timeline import get_timeline_interface  # noqa: E402

# 深度 RP/annotator 在 play 前创建(昨晚 1261 帧验证过此路径出有效深度)
import omni.replicator.core as rep  # noqa: E402
RP_D = rep.create.render_product("/Root/ur5/wrist_3_link/WristCamera/Cam", (1280, 720))
ANN_DEPTH = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
ANN_DEPTH.attach([RP_D])
try:
    _rclpy.init()
except Exception:
    pass
DEPTH_NODE = _Node("sim_depth_publisher")
DEPTH_PUB = DEPTH_NODE.create_publisher(_Image, "/depth", 10)
print("[depth] publisher ready (pre-play annotator)", flush=True)

get_timeline_interface().play()
print("=== PLAYING ===", flush=True)

import omni.graph.core as og  # noqa: E402

# 相机发布用运行时 OmniGraph(OmniLRS _create_rt_camera_graph 同款):
# 启动期挂 annotator 与 world.step 冲突会崩图执行引擎,故延迟到 play 后第 GRAPH_AT 帧建图。
# 深度例外:ROS2CameraHelper 的 depth 类型在本环境输出无效(共享RP全零/独立RP无发布),
# 改为图就绪后用 rep annotator 读回 + rclpy 发布(见 start_depth_publisher)。
GRAPH_AT = int(os.environ.get("GRAPH_AT", "900"))
graph_made = False
graph = None
ANN_DEPTH = None
DEPTH_NODE = None
DEPTH_PUB = None


def make_camera_graph():
    import usdrt.Sdf as usdrt  # noqa: E402

    k = og.Controller.Keys
    (g, _, _, _) = og.Controller.edit(
        {"graph_path": "/Root/ROS_Runtime", "evaluator_name": "push"},
        {
            k.CREATE_NODES: [
                ("OnTick", "omni.graph.action.OnTick"),
                ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                ("RPWrist", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                ("CamRGB", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                ("RPObs", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                ("CamObs", "isaacsim.ros2.bridge.ROS2CameraHelper"),
            ],
            k.SET_VALUES: [
                ("OnTick.inputs:onlyPlayback", False),
                ("RPWrist.inputs:cameraPrim", [usdrt.Path("/Root/ur5/wrist_3_link/WristCamera/Cam")]),
                ("RPWrist.inputs:width", 1280),
                ("RPWrist.inputs:height", 720),
                ("CamRGB.inputs:topicName", "/rgb"),
                ("CamRGB.inputs:type", "rgb"),
                ("CamRGB.inputs:frameId", "wrist_camera"),
                ("RPObs.inputs:cameraPrim", [usdrt.Path("/Root/ObserverCamera/Cam")]),
                ("RPObs.inputs:width", 640),
                ("RPObs.inputs:height", 360),
                ("CamObs.inputs:topicName", "/observer_rgb"),
                ("CamObs.inputs:type", "rgb"),
                ("CamObs.inputs:frameId", "observer_camera"),
            ],
            k.CONNECT: [
                ("OnTick.outputs:tick", "RPWrist.inputs:execIn"),
                ("RPWrist.outputs:execOut", "CamRGB.inputs:execIn"),
                ("RPWrist.outputs:renderProductPath", "CamRGB.inputs:renderProductPath"),
                ("OnTick.outputs:tick", "RPObs.inputs:execIn"),
                ("RPObs.outputs:execOut", "CamObs.inputs:execIn"),
                ("RPObs.outputs:renderProductPath", "CamObs.inputs:renderProductPath"),
                ("Context.outputs:context", "CamRGB.inputs:context"),
                ("Context.outputs:context", "CamObs.inputs:context"),
            ],
        },
    )
    for _ in range(5):
        og.Controller.evaluate_sync(g)
        app.update()
    print("[rt-graph] runtime camera graph ready", flush=True)
    return g




t0 = time.time()
n = 0
while time.time() - t0 < float(os.environ.get("SIM_DURATION", "2700")):
    app.update()
    n += 1
    if n == GRAPH_AT and not graph_made:
        try:
            graph = make_camera_graph()
        except Exception as e:
            print("[rt-graph] FAILED:", e, flush=True)
        graph_made = True
    if graph_made and ANN_DEPTH is not None and n % 4 == 0:
        try:
            d = np.asarray(ANN_DEPTH.get_data())
            if n % 600 == 4:
                print('[depth] get_data ->', getattr(d, 'shape', d), flush=True)
            if d.ndim == 2 and d.size:
                msg = _Image()
                msg.header.stamp = DEPTH_NODE.get_clock().now().to_msg()
                msg.height, msg.width = d.shape[0], d.shape[1]
                msg.encoding = "32FC1"
                msg.step = d.shape[1] * 4
                msg.data = np.ascontiguousarray(d, dtype=np.float32).tobytes()
                DEPTH_PUB.publish(msg)
        except Exception as e:
            print('[depth] publish error:', repr(e)[:200], flush=True)
    if n % 600 == 0:
        print(f"sim running {time.time() - t0:.0f}s, {n} frames, graph={graph_made}", flush=True)

get_timeline_interface().stop()
print("=== SIM DONE ===", flush=True)
app.close()
