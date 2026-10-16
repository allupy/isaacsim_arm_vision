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

ctx = omni.usd.get_context()
ctx.open_stage(os.environ.get("STAGE_FILE", "/ws/workbench.usd"), None)
print("=== STAGE OPENED ===", flush=True)

from omni.timeline import get_timeline_interface  # noqa: E402

get_timeline_interface().play()
print("=== PLAYING ===", flush=True)

import omni.graph.core as og  # noqa: E402

# 相机发布改用运行时 OmniGraph(OmniLRS simulation_manager_ros2._create_rt_camera_graph 同款):
# annotator 手动读回与 world.step 冲突会导致图执行引擎崩溃/卡死,
# 故延迟到 play 后第 GRAPH_AT 帧再建图, 由 ROS2CameraHelper 在图管线内原生发布。
GRAPH_AT = int(os.environ.get("GRAPH_AT", "900"))
graph_made = False
graph = None


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
                ("CamDepth", "isaacsim.ros2.bridge.ROS2CameraHelper"),
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
                ("CamDepth.inputs:topicName", "/depth"),
                ("CamDepth.inputs:type", "depth"),
                ("CamDepth.inputs:frameId", "wrist_camera"),
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
                ("RPWrist.outputs:execOut", "CamDepth.inputs:execIn"),
                ("RPWrist.outputs:renderProductPath", "CamDepth.inputs:renderProductPath"),
                ("OnTick.outputs:tick", "RPObs.inputs:execIn"),
                ("RPObs.outputs:execOut", "CamObs.inputs:execIn"),
                ("RPObs.outputs:renderProductPath", "CamObs.inputs:renderProductPath"),
                ("Context.outputs:context", "CamRGB.inputs:context"),
                ("Context.outputs:context", "CamDepth.inputs:context"),
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
    if n % 600 == 0:
        print(f"sim running {time.time() - t0:.0f}s, {n} frames, graph={graph_made}", flush=True)

get_timeline_interface().stop()
print("=== SIM DONE ===", flush=True)
app.close()
