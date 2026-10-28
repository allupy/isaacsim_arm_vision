#!/usr/bin/env python3
"""无头运行 workbench 场景(以引用方式组合——本机验证过可渲染):

- 空舞台 + AddReference(workbench.usd) + TestCam 式自建相机渲染已被验证可行
- /rgb /depth /observer_rgb 由本进程直接用 rclpy 发布(annotator 读数),
  绕开 ROS2CameraHelper(在本环境不稳定)
- 旧 ROS_JointStates 图从实验副本移除;关节状态和命令通过 PhysX articulation + rclpy 桥接
"""
import os
import time
import json
from pathlib import Path
import faulthandler
import sys
import array
sys.path.insert(0, '/experiment/src')
from modules.trajectory_playback import PhysicsTrajectory
from modules.video_capture import ExperimentVideoCapture
runtime_path = Path('/experiment/sim/runtime_config.json')
RUNTIME = json.loads(runtime_path.read_text()) if runtime_path.exists() else {}
OPTIMIZED = RUNTIME.get('optimized', True)
EMBEDDED = RUNTIME.get('embedded_trajectory', OPTIMIZED)
WRIST_SIZE = tuple(RUNTIME.get('wrist_resolution', [640,360] if OPTIMIZED else [1280,720]))
OBS_SIZE = (640,360)
faulthandler.enable()
faulthandler.dump_traceback_later(60, repeat=True)

from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "renderer": "RayTracedLighting",
                     "limit_cpu_threads": 8, "multi_gpu": False,
                     "create_new_stage": False, "disable_viewport_updates": True})

import carb  # noqa: E402
import omni.usd  # noqa: E402
import omni.kit.app  # noqa: E402

for key in ("/physics/enforceRealtime", "/app/physics/enforceRealtime"):
    try:
        carb.settings.get_settings().set(key, True)
    except Exception:
        pass

from pxr import Usd, UsdGeom, Gf
# Remove legacy OmniGraph nodes before attaching the stage to Kit. Opening the
# original graph with the bridge enabled can block stage activation indefinitely.
clean_stage = Usd.Stage.Open('/ws/workbench.usd')
for path in [str(p.GetPath()) for p in clean_stage.Traverse() if p.GetTypeName() == 'OmniGraph']:
    clean_stage.RemovePrim(path)
if clean_stage.GetPrimAtPath('/Render'):
    clean_stage.RemovePrim('/Render')
Path('/experiment/sim').mkdir(exist_ok=True)
clean_path=f'/experiment/sim/direct_bridge_scene_{time.time_ns()}.usdc'
clean_stage.GetRootLayer().Export(clean_path)
del clean_stage
ctx = omni.usd.get_context()
ctx.open_stage(clean_path, None)
stage = ctx.get_stage()
faulthandler.cancel_dump_traceback_later()
omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(
    'isaacsim.ros2.bridge', True)
authored_joint_positions = {}
for prim in stage.Traverse():
    for attr in prim.GetAttributes():
        if attr.GetName() == 'inputs:domain_id' or attr.GetName() == 'inputs:domainId':
            attr.Set(int(os.environ.get('ROS_DOMAIN_ID', '78')))
        if attr.GetName() == 'inputs:useDomainIDEnvVar':
            attr.Set(True)
    # Preserve the authored parked pose instead of driving every joint to zero
    # on Play. The source scene has nonzero joint states but zero drive targets.
    if os.environ.get('REPAIR_PARKED_POSE', '1') == '1' and prim.GetTypeName() == 'PhysicsRevoluteJoint':
        initial = prim.GetAttribute('state:angular:physics:position').Get()
        if initial is not None:
            authored_joint_positions[prim.GetName()] = float(initial)*3.141592653589793/180.
        target = prim.GetAttribute('drive:angular:physics:targetPosition')
        if initial is not None and target.IsValid():
            target.Set(float(initial))
    if prim.GetName() == 'PublisherJointState':
        prim.GetAttribute('inputs:topicName').Set('/sim_joint_states')
OUT = Path('/experiment/sim')
OUT.mkdir(exist_ok=True)
video_capture=ExperimentVideoCapture(OUT,Path('/experiment/videos'))
def world_matrix(path):
    return np.asarray(UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T

def camera_spec(path, width, height):
    cam = UsdGeom.Camera(stage.GetPrimAtPath(path))
    fx = width * cam.GetFocalLengthAttr().Get() / cam.GetHorizontalApertureAttr().Get()
    fy = height * cam.GetFocalLengthAttr().Get() / cam.GetVerticalApertureAttr().Get()
    matrix = world_matrix(path) @ np.diag([1., -1., -1., 1.])
    return {'K': [[fx, 0., width/2], [0., fy, height/2], [0., 0., 1.]], 'optical_to_world': matrix.tolist()}

# 场景坐标按米编写,但 stage 的 metersPerUnit=0.01(1单位=1cm),
# depth annotator 会按 metersPerUnit 再折算一次,导致输出深度只有真值的 1/100。
# 这里按 1/metersPerUnit 还原为米,不硬编码。
from pxr import UsdGeom as _UsdGeom
DEPTH_UNIT_SCALE = 1.0 / (_UsdGeom.GetStageMetersPerUnit(stage) or 1.0)
print(f"depth unit scale = {DEPTH_UNIT_SCALE}", flush=True)
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

# Restore a documented eye-in-hand mount in this in-memory experimental scene.
# The source USD and its authored camera poses remain unchanged on disk.
def repair_camera_mounts():
    R_ee_optical = np.array([[.999999699,-.000770058,-.000097541],
                         [.000769941,.999998993,-.001192037],
                         [.000098459,.001191961,.999999285]])
    T = np.eye(4)
    T[:3,:3] = R_ee_optical @ np.diag([1.,-1.,-1.])
    T[:3,3] = [.012674,-.116914,-.000172]
    camera_mount = UsdGeom.Xformable(stage.GetPrimAtPath('/Root/ur5/wrist_3_link/WristCamera'))
    camera_mount.ClearXformOpOrder()
    camera_mount.AddTransformOp().Set(Gf.Matrix4d(T.T.tolist()))
    camera_local = UsdGeom.Xformable(stage.GetPrimAtPath('/Root/ur5/wrist_3_link/WristCamera/Cam'))
    camera_local.ClearXformOpOrder()
    observer_mount = UsdGeom.Xformable(stage.GetPrimAtPath('/Root/ObserverCamera'))
    observer_mount.ClearXformOpOrder()
    observer_mount.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(1.3,-1.1,1.1),
                                                         Gf.Vec3d(.45,0.,.15),Gf.Vec3d(0.,0.,1.)).GetInverse())
    UsdGeom.Xformable(stage.GetPrimAtPath('/Root/ObserverCamera/Cam')).ClearXformOpOrder()

if os.environ.get('REPAIR_CAMERA_MOUNT', '0') == '1':
    repair_camera_mounts()

GRIPPER_PROFILE=dict(id='robotiq_2f140',label='ORIGINAL ROBOTIQ 2F-140',tcp_offset_m=.185,command_scale=1.)
if RUNTIME.get('gripper_profile','robotiq_2f140')=='short_parallel':
    from modules.short_parallel_gripper import build_short_parallel_gripper
    GRIPPER_PROFILE=build_short_parallel_gripper(stage)
    # The old revolute master shares this name; its authored angle must never
    # be replayed as a displacement on the replacement prismatic joint.
    authored_joint_positions['finger_joint']=0.
    authored_joint_positions['right_finger_joint']=0.
print('GRIPPER_PROFILE '+json.dumps(GRIPPER_PROFILE),flush=True)
OBJECT_NAMES=['red_cube','green_cube','blue_cube']
DESTINATIONS={}
if RUNTIME.get('demo_scene')=='everyday':
    from modules.demo_scene import add_everyday_scene
    OBJECT_NAMES=add_everyday_scene(stage)
    if RUNTIME.get('physical_destinations'):
        from modules.destination_scene import build_destinations
        DESTINATIONS=build_destinations(stage)

RP_WRIST = rep.create.render_product("/Root/ur5/wrist_3_link/WristCamera/Cam", WRIST_SIZE)
RP_OBS = rep.create.render_product("/Root/ObserverCamera/Cam", OBS_SIZE)
ANN_RGB = rep.AnnotatorRegistry.get_annotator("rgb")
ANN_DEPTH = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
ANN_OBS = rep.AnnotatorRegistry.get_annotator("rgb")
ANN_OBS_DEPTH = rep.AnnotatorRegistry.get_annotator('distance_to_image_plane')
ANN_RGB.attach(RP_WRIST)
ANN_DEPTH.attach(RP_WRIST)
ANN_OBS.attach(RP_OBS)
ANN_OBS_DEPTH.attach(RP_OBS)
print("annotators attached", flush=True)

import rclpy  # noqa: E402
from rclpy.node import Node  # noqa: E402
from sensor_msgs.msg import Image  # noqa: E402
from sensor_msgs.msg import JointState
from rosgraph_msgs.msg import Clock
from std_msgs.msg import String

try:
    rclpy.init()
except Exception:
    pass  # bridge 可能已初始化
ros_node = Node("sim_camera_publisher")
pub_rgb = ros_node.create_publisher(Image, "/rgb", 10)
pub_depth = ros_node.create_publisher(Image, "/depth", 10)
pub_obs = ros_node.create_publisher(Image, "/observer_rgb", 10)
pub_obs_depth = ros_node.create_publisher(Image, '/observer_depth', 10)
pub_calib = ros_node.create_publisher(String, '/feedback/camera_calibration', 10)
pub_truth = ros_node.create_publisher(String, '/evaluation/ground_truth', 10)
pub_arm_joints = ros_node.create_publisher(JointState, '/joint_states', 10)
pub_full_joints = ros_node.create_publisher(JointState, '/feedback/joint_states', 10)
pub_clock = ros_node.create_publisher(Clock, '/clock', 1)
ARM_NAMES = ['shoulder_pan_joint','shoulder_lift_joint','elbow_joint',
             'wrist_1_joint','wrist_2_joint','wrist_3_joint']

def publish_joint_feedback():
    # MoveIt only receives joints present in its URDF. Retain gripper feedback
    # on a separate channel, and stamp both from the same measured sample.
    msg = JointState()
    msg.header.stamp = ros_node.get_clock().now().to_msg()
    msg.name = joint_names
    msg.position = np.asarray(arm.get_joint_positions(),dtype=float).tolist()
    msg.velocity = np.asarray(arm.get_joint_velocities(),dtype=float).tolist()
    if GRIPPER_PROFILE['command_scale']!=1. and 'finger_joint' in msg.name:
        i=msg.name.index('finger_joint')
        msg.position[i]/=GRIPPER_PROFILE['command_scale']
        msg.velocity[i]/=GRIPPER_PROFILE['command_scale']
    pub_full_joints.publish(msg)
    if not all(name in msg.name for name in ARM_NAMES):
        return
    filtered = JointState()
    filtered.header = msg.header
    filtered.name = ARM_NAMES
    indices = [msg.name.index(name) for name in ARM_NAMES]
    filtered.position = [msg.position[i] for i in indices]
    if len(msg.velocity) == len(msg.name):
        filtered.velocity = [msg.velocity[i] for i in indices]
    pub_arm_joints.publish(filtered)

def command_callback(msg):
    if len(msg.name) != len(msg.position) or not np.isfinite(msg.position).all():
        print('REJECT_COMMAND malformed',flush=True)
        return
    if not all(name in joint_names for name in msg.name):
        print('REJECT_COMMAND unknown joint',flush=True)
        return
    indices = np.array([joint_names.index(name) for name in msg.name],dtype=np.int32)
    values = np.asarray(msg.position)
    if np.any(values < joint_limits['lower'][indices]-.001) or np.any(values > joint_limits['upper'][indices]+.001):
        print('REJECT_COMMAND joint limit '+str(dict(zip(msg.name,msg.position))),flush=True)
        return
    arm.apply_action(ArticulationAction(joint_positions=values,joint_indices=indices))
    command_updates.append([sim.current_time,time.monotonic(),list(msg.name),values.tolist()])

active_trajectory = None
execution_samples = []
execution_stream = (OUT/'execution.jsonl').open('w')
command_updates = []
command_stream = (OUT/'command_updates.jsonl').open('w')
camera_requested = False
request_seen = False

def camera_request_callback(msg):
    global camera_requested, request_seen
    camera_requested = True
    request_seen = True

camera_request_sub = ros_node.create_subscription(String,'/feedback/camera_request',camera_request_callback,1)

def trajectory_callback(msg):
    global active_trajectory, execution_samples
    try:
        limits = {name:(joint_limits['lower'][i],joint_limits['upper'][i]) for i,name in enumerate(joint_names)}
        payload = json.loads(msg.data)
        # Keep the controller's single logical 0..0.78 open/close API; the short
        # gripper's master DOF is a physical prismatic displacement (profile scale).
        if GRIPPER_PROFILE['command_scale']!=1. and 'finger_joint' in payload['names']:
            i=payload['names'].index('finger_joint')
            for q in payload['positions']:
                q[i]*=GRIPPER_PROFILE['command_scale']
            for v in payload.get('velocities',[]):
                v[i]*=GRIPPER_PROFILE['command_scale']
        candidate = PhysicsTrajectory(payload['names'],payload['times'],payload['positions'],sim.current_time,limits,
                                      velocities=payload.get('velocities'))
        # Only a validated new trajectory replaces the previous one.
        active_trajectory = candidate
        execution_samples = []
    except (ValueError,KeyError,TypeError) as exc:
        print('REJECT_TRAJECTORY '+str(exc),flush=True)

if EMBEDDED:
    # Isaac's bundled ROS runtime lacks trajectory_msgs; use a validated JSON envelope.
    command_sub = ros_node.create_subscription(String, '/feedback/timed_trajectory', trajectory_callback, 1)
else:
    command_sub = ros_node.create_subscription(JointState, '/joint_command', command_callback, 10)
print("rclpy publishers ready", flush=True)


def to_img_msg(data, encoding, stamp):
    msg = Image()
    msg.header.stamp = stamp
    msg.height, msg.width = data.shape[0], data.shape[1]
    msg.encoding = encoding
    msg.step = data.strides[0]
    # Generated ROS setters validate plain bytes element by element in Python.
    # A typed byte array follows the checked buffer fast path, preserving every byte.
    msg.data = array.array('B',data.tobytes()) if OPTIMIZED else data.tobytes()
    return msg


def publish_cameras():
    """读取 annotator 并发布(数据未就绪/为空则跳过)"""
    stamp = ros_node.get_clock().now().to_msg()
    rgb = np.asarray(ANN_RGB.get_data())
    if rgb.ndim == 3 and rgb.size:
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        pub_rgb.publish(to_img_msg(rgb[:, :, ::-1].copy(), "bgr8", stamp))
    d = np.asarray(ANN_DEPTH.get_data())
    if d.ndim == 2 and d.size:
        pub_depth.publish(to_img_msg(np.ascontiguousarray(d * DEPTH_UNIT_SCALE, dtype=np.float32), "32FC1", stamp))
    obs = np.asarray(ANN_OBS.get_data())
    if obs.ndim == 3 and obs.size:
        if obs.shape[-1] == 4:
            obs = obs[..., :3]
        pub_obs.publish(to_img_msg(obs[:, :, ::-1].copy(), "bgr8", stamp))
    obs_depth = np.asarray(ANN_OBS_DEPTH.get_data())
    if obs_depth.ndim == 2 and obs_depth.size:
        pub_obs_depth.publish(to_img_msg(np.ascontiguousarray(obs_depth * DEPTH_UNIT_SCALE, dtype=np.float32), '32FC1', stamp))
    stamp_ns = stamp.sec * 1000000000 + stamp.nanosec
    calib = {'stamp_ns': stamp_ns, 'wrist': camera_spec('/Root/ur5/wrist_3_link/WristCamera/Cam', *WRIST_SIZE),
             'observer': camera_spec('/Root/ObserverCamera/Cam', 640, 360),
             'planning_to_world': world_matrix('/Root/ur5/base_link').tolist(),
             'gripper':GRIPPER_PROFILE}
    pub_calib.publish(String(data=json.dumps(calib)))
    truth = {'stamp_ns': stamp_ns, 'sim_time': sim.current_time, 'objects': {name: world_matrix('/Root/' + name)[:3, 3].tolist()
             for name in OBJECT_NAMES},'destinations':DESTINATIONS,
             'wrist': world_matrix('/Root/ur5/wrist_3_link').tolist(),
             'gripper':GRIPPER_PROFILE,
             'finger_positions':{name:float(arm.get_joint_positions()[joint_names.index(name)])
                 for name in joint_names if 'finger' in name or 'knuckle' in name},
             'base': world_matrix('/Root/ur5/base_link').tolist()}
    pub_truth.publish(String(data=json.dumps(truth)))
    if cam_frames % 25 == 0:
        import cv2
        if rgb.ndim == 3 and rgb.size:
            cv2.imwrite(str(OUT/'wrist.png'), rgb[..., :3][:, :, ::-1].copy())
        if obs.ndim == 3 and obs.size:
            cv2.imwrite(str(OUT/'observer.png'), obs[..., :3][:, :, ::-1].copy())
        (OUT/'latest.json').write_text(json.dumps({'truth': truth, 'calibration': calib}, indent=2))
        print(f'SENSOR frame={cam_frames} depth={float(np.nanmedian(d)):.4f} objects={truth["objects"]}', flush=True)


# ------------------------------------------------------------------
# 播放循环
# ------------------------------------------------------------------
DURATION = float(os.environ.get("SIM_DURATION", "180"))

from omni.timeline import get_timeline_interface  # noqa: E402
timeline = get_timeline_interface()
from isaacsim.core.api import SimulationContext
physics_path = next(str(p.GetPath()) for p in stage.Traverse() if p.GetTypeName() == 'PhysicsScene')
sim = SimulationContext(physics_dt=1/120,rendering_dt=1/60,
                        stage_units_in_meters=UsdGeom.GetStageMetersPerUnit(stage),
                        physics_prim_path=physics_path,set_defaults=False,device='cpu')
sim.initialize_physics()
sim.play()
from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
from isaacsim.core.utils.types import ArticulationAction
arm = SingleArticulation('/Root/ur5',name='feedback_arm',reset_xform_properties=False)
arm.initialize()
joint_names = list(arm.dof_names)
joint_limits = arm.dof_properties
reset_objects = {}
for name in OBJECT_NAMES:
    body = SingleRigidPrim('/Root/'+name,name='reset_'+name,reset_xform_properties=False)
    body.initialize()
    reset_objects[name] = (body, body.get_world_pose())
print('JOINT_LIMITS '+str(joint_limits),flush=True)
print("=== PLAYING ===", flush=True)

CAM_EVERY = int(os.environ.get("CAM_EVERY", "4"))  # 每 N 帧发布一次相机
t0 = time.time()
n = 0
cam_frames = 0
last_render = 0.
last_capture = 0.
profile_start = time.monotonic()
physics_wall = render_wall = camera_wall = 0.
physics_count = render_count = 0
steps_per_batch = int(RUNTIME.get('physics_batch', 4 if OPTIMIZED else 120))
render_period = float(RUNTIME.get('render_period', .10 if OPTIMIZED else 0.))
camera_period = float(RUNTIME.get('camera_period', .20))
print('RUNTIME_CONFIG '+json.dumps(dict(optimized=OPTIMIZED,embedded=EMBEDDED,wrist=WRIST_SIZE,
                                       physics_batch=steps_per_batch,render_period=render_period)),flush=True)
while time.time() - t0 < DURATION:
    perturb_request=OUT/'perturb_request.json'
    if perturb_request.exists():
        request=json.loads(perturb_request.read_text())
        offset=np.asarray(request['offset'],dtype=float)
        if offset.shape!=(3,) or not np.all(np.isfinite(offset)) or np.linalg.norm(offset)>.05:
            raise ValueError('Perturbation fixture limited to 5 cm')
        body,_=reset_objects[request['object']]
        position,orientation=body.get_world_pose()
        body.set_world_pose(position=position+offset,orientation=orientation)
        body.set_linear_velocity(np.zeros(3));body.set_angular_velocity(np.zeros(3))
        perturb_request.unlink()
        (OUT/'perturb_ack.txt').write_text(request['id'])
        print('EXPERIMENT_PERTURBATION '+json.dumps(request),flush=True)
    reset_request = Path('/experiment/sim/reset_scene_requested')
    if reset_request.exists():
        reset_id = reset_request.read_text().strip()
        active_trajectory = None
        execution_samples = []
        command_updates.clear()
        command_stream.close()
        command_stream = (OUT/'command_updates.jsonl').open('w')
        execution_stream.close()
        execution_stream = (OUT/'execution.jsonl').open('w')
        reset_q = np.asarray(arm.get_joint_positions(),dtype=float)
        for i,name in enumerate(joint_names):
            reset_q[i] = authored_joint_positions.get(name,0.)
        arm.set_joint_positions(reset_q)
        arm.set_joint_velocities(np.zeros(len(joint_names)))
        arm.apply_action(ArticulationAction(joint_positions=reset_q))
        for body,(position,orientation) in reset_objects.values():
            body.set_world_pose(position=position,orientation=orientation)
            body.set_linear_velocity(np.zeros(3))
            body.set_angular_velocity(np.zeros(3))
        reset_request.unlink()
        Path('/experiment/sim/reset_scene_ack').write_text(reset_id)
        print('SCENE_RESET_COMPLETE '+reset_id,flush=True)
    if Path('/experiment/sim/reset_arm_requested').exists():
        Path('/experiment/sim/reset_arm_requested').unlink()
        reset_q = np.asarray(arm.get_joint_positions(),dtype=float)
        for i,name in enumerate(joint_names):
            if name in authored_joint_positions:
                reset_q[i] = authored_joint_positions[name]
        arm.set_joint_positions(reset_q)
        arm.set_joint_velocities(np.zeros(len(joint_names)))
        arm.apply_action(ArticulationAction(joint_positions=reset_q))
        print('ARM_RESET_COMPLETE '+str(sim.current_time),flush=True)
    phase_start = time.monotonic()
    for _ in range(steps_per_batch):
        if active_trajectory is not None:
            values, done = active_trajectory.sample(sim.current_time)
            indices = np.array([joint_names.index(name) for name in active_trajectory.names],dtype=np.int32)
            arm.apply_action(ArticulationAction(joint_positions=values,joint_indices=indices))
            command_updates.append([sim.current_time,time.monotonic(),active_trajectory.names,values.tolist()])
            execution_samples.append([sim.current_time, time.monotonic(), values.tolist(),
                                      np.asarray(arm.get_joint_positions())[indices].tolist()])
            if done:
                execution_stream.write(json.dumps(dict(names=active_trajectory.names,
                    duration=active_trajectory.times[-1],samples=execution_samples))+'\n')
                execution_stream.flush()
                active_trajectory = None
        sim.step(render=False)
        physics_count += 1
        seconds = sim.current_time
        clock = Clock()
        clock.clock.sec = int(seconds)
        clock.clock.nanosec = int((seconds-int(seconds))*1e9)
        pub_clock.publish(clock)
        rclpy.spin_once(ros_node,timeout_sec=0.)
        if sim.current_time_step_index % 4 == 0:
            publish_joint_feedback()
    physics_wall += time.monotonic()-phase_start
    video_capture.requests()
    if not video_capture.active and time.monotonic()-last_render < render_period:
        continue
    phase_start = time.monotonic()
    sim.render()
    render_wall += time.monotonic()-phase_start
    last_render = time.monotonic()
    render_count += 1
    if video_capture.active:
        video_capture.capture(np.asarray(ANN_OBS.get_data()),np.asarray(ANN_RGB.get_data()),sim.current_time)
    for update in command_updates:
        command_stream.write(json.dumps(update)+'\n')
    command_stream.flush()
    command_updates.clear()
    for _ in range(3):
        rclpy.spin_once(ros_node,timeout_sec=0.)
    n += 1
    if (not OPTIMIZED and n % CAM_EVERY == 0) or (OPTIMIZED and
            (camera_requested or time.monotonic()-last_capture > camera_period)):
        phase_start = time.monotonic()
        publish_cameras()
        camera_wall += time.monotonic()-phase_start
        cam_frames += 1
        camera_requested = False
        last_capture = time.monotonic()
    if render_count % 50 == 0:
        elapsed = time.monotonic()-profile_start
        (OUT/'performance.json').write_text(json.dumps(dict(wall_s=elapsed,sim_s=sim.current_time,
            physics_wall_s=physics_wall,render_wall_s=render_wall,camera_wall_s=camera_wall,
            physics_steps=physics_count,renders=render_count,render_fps=render_count/elapsed),indent=2))
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
if video_capture.active:
    video_capture.finish()
print(f"=== SIM DONE (cam_pub={cam_frames}) ===", flush=True)
app.close()
