#!/usr/bin/env python3
"""
在 ur5_depth.usd 基础上搭建实验室工作台场景,保存 workbench.usd:

1. 移除失效的 checkerboard 引用
2. 添加 地面 + 工作台桌板 + 灯光
3. 在桌面放置 3 个彩色方块(红/绿/蓝,5cm,随机朝向,刚体+碰撞)
4. 按手眼标定值在腕部(wrist_3_link 下)挂载 D455 内参的相机
5. 创建 ROS2 相机发布图:/rgb(腕部) /depth(腕部) /observer_rgb(全局观察相机)
6. 打印机械臂基座位置与已有 ROS 图的话题名
"""
from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import omni.usd
import omni.kit.app
from pxr import Usd, UsdGeom, UsdPhysics, UsdLux, UsdShade, Gf, Sdf
import random

random.seed(42)

SRC = "/ws/ur5_depth.usd"
DST = "/ws/workbench.usd"

ctx = omni.usd.get_context()
ctx.open_stage(SRC, None)
stage = ctx.get_stage()
print("=== STAGE OPENED ===")

# ------------------------------------------------------------------
# 0. 打印基座位置 & 已有图的话题
# ------------------------------------------------------------------
ur5_prim = stage.GetPrimAtPath("/Root/ur5")
base_tf = UsdGeom.Xformable(ur5_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()) if ur5_prim and ur5_prim.IsActive() else None
base_pos = Gf.Vec3d(base_tf[3][0], base_tf[3][1], base_tf[3][2]) if base_tf else Gf.Vec3d(0, 0, 0)
print(f"ARM_BASE_POS = ({base_pos[0]:.3f}, {base_pos[1]:.3f}, {base_pos[2]:.3f})")

import omni.graph.core as og
for graph_path in ["/Root/ur5/ROS_JointStates"]:
    try:
        g = og.get_graph_by_path(graph_path)
        if g:
            for node in g.get_nodes():
                attrs = {}
                for a in node.get_attributes():
                    name = a.get_name()
                    if any(k in name.lower() for k in ("topic", "frame")):
                        try:
                            attrs[name] = a.get()
                        except Exception:
                            pass
                if attrs:
                    print(f"GRAPH_NODE {node.get_type_name()} {attrs}")
    except Exception as e:
        print("graph inspect error:", e)

# ------------------------------------------------------------------
# 1. 清理:陈旧渲染产品、遗留图(含原场景失效的腕部相机图)、失效引用
# ------------------------------------------------------------------
if stage.GetPrimAtPath("/Render").IsValid():
    stage.RemovePrim("/Render")
    print("removed stale /Render render products")

# 先收集全部图路径再移除(边遍历边删会使节点 prim 失效导致崩溃)
graph_paths = []
for prim in stage.Traverse():
    try:
        if prim.IsValid() and prim.GetTypeName() == "OmniGraph":
            graph_paths.append(str(prim.GetPath()))
    except Exception:
        pass

removed_graphs = []
for gp in graph_paths:
    if gp != "/Root/ur5/ROS_JointStates":
        try:
            stage.RemovePrim(gp)
            removed_graphs.append(gp)
        except Exception as e:
            print(f"graph remove failed {gp}: {e}")
print("all graphs found:", graph_paths)
print("removed legacy graphs:", removed_graphs)

legacy_cams = []
for prim in stage.Traverse():
    try:
        if prim.IsValid() and prim.GetTypeName() == "Camera":
            p = str(prim.GetPath())
            if "WristCamera" not in p and "ObserverCamera" not in p:
                legacy_cams.append(p)
    except Exception:
        pass
print("camera prims in stage (legacy kept as-is):", legacy_cams)

if stage.GetPrimAtPath("/Root/checkerboard_6x10").IsValid():
    stage.RemovePrim("/Root/checkerboard_6x10")
    print("removed broken checkerboard")

# ------------------------------------------------------------------
# 2. 地面 + 桌台 + 灯光
# ------------------------------------------------------------------
base_z = base_pos[2]  # 桌面高度 = 机械臂基座高度(机械臂栓在桌上)

# 地面(24x24m 薄板:注意 Cube 的 size 是边长,必须用 scale 压扁,
# 顶面 z=0。之前误用 size=12 造成 12m 实心巨块把整个场景埋住——相机全黑的根因)
ground = UsdGeom.Cube.Define(stage, "/Root/Ground")
ground.CreateSizeAttr(1.0)
ground.AddScaleOp().Set(Gf.Vec3f(24.0, 24.0, 0.1))
ground.AddTranslateOp().Set(Gf.Vec3f(0.25, 0.0, -0.05))
UsdPhysics.CollisionAPI.Apply(ground.GetPrim())

# 桌板(顶面 z = base_z)
table = UsdGeom.Cube.Define(stage, "/Root/Table")
table.CreateSizeAttr(1.0)
table.AddScaleOp().Set(Gf.Vec3f(1.6, 1.6, 0.02))  # 1.6x1.6m 板
table.AddTranslateOp().Set(Gf.Vec3f(0.25, 0.0, base_z - 0.01))
UsdPhysics.CollisionAPI.Apply(table.GetPrim())
print(f"table top at z={base_z}")

# 桌腿(视觉用,4根)
for i, (dx, dy) in enumerate([(0.8, 0.8), (-0.8, 0.8), (0.8, -0.8), (-0.8, -0.8)]):
    leg = UsdGeom.Cube.Define(stage, f"/Root/TableLeg{i}")
    leg.CreateSizeAttr(1.0)
    leg.AddScaleOp().Set(Gf.Vec3f(0.06, 0.06, base_z / 2))
    leg.AddTranslateOp().Set(Gf.Vec3f(0.25 + dx, dy, base_z / 2))

# 灯光
dome = UsdLux.DomeLight.Define(stage, "/Root/DomeLight")
dome.CreateIntensityAttr(1500.0)
dist = UsdLux.DistantLight.Define(stage, "/Root/WorkLight")
dist.CreateIntensityAttr(2500.0)
dist.AddRotateXOp().Set(-45.0)

# 物理场景(原 USD 没有,headless play 需要显式定义)
phys_scene = UsdPhysics.Scene.Define(stage, "/Root/PhysicsScene")
phys_scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
phys_scene.CreateGravityMagnitudeAttr().Set(9.81)

# ------------------------------------------------------------------
# 3. 彩色方块
# ------------------------------------------------------------------
CUBE_SIZE = 0.05
CUBES = [
    ("red_cube", Gf.Vec3d(0.40, 0.18, 0), Gf.Vec3f(0.8, 0.1, 0.1)),
    ("green_cube", Gf.Vec3d(0.48, -0.05, 0), Gf.Vec3f(0.1, 0.8, 0.1)),
    ("blue_cube", Gf.Vec3d(0.33, -0.22, 0), Gf.Vec3f(0.1, 0.2, 0.9)),
]

for name, xy, color in CUBES:
    cube = UsdGeom.Cube.Define(stage, f"/Root/{name}")
    cube.CreateSizeAttr(CUBE_SIZE)
    yaw = random.uniform(0, 180)
    xfo = UsdGeom.XformCommonAPI(cube)
    xfo.SetTranslate(Gf.Vec3d(xy[0], xy[1], base_z + CUBE_SIZE / 2 + 0.02))
    xfo.SetRotate(Gf.Vec3f(0, 0, yaw))
    UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    UsdPhysics.MassAPI.Apply(cube.GetPrim()).CreateMassAttr(0.1)
    print(f"cube {name} at ({xy[0]:.2f},{xy[1]:.2f},{base_z:.2f}) yaw={yaw:.0f}")

# ------------------------------------------------------------------
# 4. 材质(纯色,无贴图依赖)
# ------------------------------------------------------------------
mats_prim = stage.DefinePrim("/Root/Looks", "Scope")
for name, color in [("red_cube", CUBES[0][2]), ("green_cube", CUBES[1][2]), ("blue_cube", CUBES[2][2]),
                    ("table", Gf.Vec3f(0.55, 0.55, 0.58)), ("ground", Gf.Vec3f(0.35, 0.38, 0.35))]:
    mat = UsdShade.Material.Define(stage, f"/Root/Looks/{name}_mat")
    shader = UsdShade.Shader.Define(stage, f"/Root/Looks/{name}_mat/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "outputs:out")
    target = {"table": "/Root/Table", "ground": "/Root/Ground"}.get(name, f"/Root/{name}")
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(target)).Bind(mat)

# ------------------------------------------------------------------
# 5. 腕部相机(按手眼标定挂到 wrist_3_link)
# ------------------------------------------------------------------
# 找 wrist_3_link prim
wrist_prim = None
for prim in stage.Traverse():
    if "wrist_3_link" in prim.GetName():
        wrist_prim = prim
        break
if wrist_prim is None:
    for prim in stage.Traverse():
        if "tool0" in prim.GetName():
            wrist_prim = prim
            break
print(f"WRIST_PRIM = {wrist_prim.GetPath() if wrist_prim else None}")

# 手眼标定: T_ee_cam(optical) = [R_cam2gripper | t_cam2gripper]
import numpy as np
R_cam2gripper = np.array([
    [9.99999699e-01, -7.70058018e-04, -9.75410307e-05],
    [7.69941203e-04, 9.99998993e-01, -1.19203713e-03],
    [9.84588702e-05, 1.19196167e-03, 9.99999285e-01]])
t_cam2gripper = np.array([0.012674, -0.116914, -0.000172])

# USD 相机约定(-z 朝前) → ROS optical(+z 朝前):绕 x 轴转 180°
R_optical_from_usd = np.diag([1.0, -1.0, -1.0])
R_ee_from_usd = R_cam2gripper @ R_optical_from_usd

cam_xf = UsdGeom.Xform.Define(stage, "/Root/ur5/WristCamera")
if wrist_prim is not None:
    # 作为 wrist_3_link 的子 prim(继承其运动)
    cam_xf = UsdGeom.Xform.Define(stage, wrist_prim.GetPath().AppendPath("WristCamera"))
cam_xf.AddTransformOp().Set(
    Gf.Matrix4d(
        R_ee_from_usd[0][0], R_ee_from_usd[1][0], R_ee_from_usd[2][0], 0.0,
        R_ee_from_usd[0][1], R_ee_from_usd[1][1], R_ee_from_usd[2][1], 0.0,
        R_ee_from_usd[0][2], R_ee_from_usd[1][2], R_ee_from_usd[2][2], 0.0,
        t_cam2gripper[0], t_cam2gripper[1], t_cam2gripper[2], 1.0))

cam = UsdGeom.Camera.Define(stage, cam_xf.GetPath().AppendPath("Cam"))
# D455 内参: fx=634.17, 1280x720 → focalLength = aperture * fx / width
FX, WIDTH, HEIGHT = 634.165821, 1280.0, 720.0
aperture_h, aperture_v = 20.955, 20.955 * HEIGHT / WIDTH
cam.CreateFocalLengthAttr(aperture_h * FX / WIDTH)
cam.CreateHorizontalApertureAttr(aperture_h)
cam.CreateVerticalApertureAttr(aperture_v)
cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 10.0))
print(f"wrist camera at {cam.GetPath()} fx={FX}")

# ------------------------------------------------------------------
# 6. 全局观察相机:桌面正上方俯视(USD 相机默认朝 -z,无旋转即垂直向下,
#    已在本机验证可见场景;之前手算 lookat 矩阵行/列序反了拍到天空)
# ------------------------------------------------------------------
obs_xf = UsdGeom.Xform.Define(stage, "/Root/ObserverCamera")
obs_xf.AddTranslateOp().Set(Gf.Vec3d(0.35, 0.0, base_z + 2.0))
obs_cam = UsdGeom.Camera.Define(stage, "/Root/ObserverCamera/Cam")
obs_cam.CreateFocalLengthAttr(18.0)  # 俯视 1.6x1.6m 桌面全貌
obs_cam.CreateHorizontalApertureAttr(20.955)
obs_cam.CreateVerticalApertureAttr(20.955 * 720.0 / 1280.0)

# ------------------------------------------------------------------
# 7. 相机发布图改在 run_sim.py 运行时创建(rep.render_product,无头标准方式)
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# 8. 保存(只导出到新文件,绝不改动原 ur5_depth.usd)
# ------------------------------------------------------------------
stage.Export(DST)
print(f"SAVED {DST}")
app.close()
