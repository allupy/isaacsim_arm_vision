"""Read-only source-stage test of the authored replacement and joint anchors."""
import glob
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
for pattern in ['/isaac-sim/extscache/omni.usd.libs-*','/isaac-sim/extscache/omni.usd.schema.physx-*']:
    sys.path.insert(0,glob.glob(pattern)[0])
import numpy as np
from pxr import Plug
Plug.Registry().RegisterPlugins(glob.glob('/isaac-sim/extscache/omni.usd.schema.physx-*')[0]+'/resources')
from pxr import Usd,UsdGeom,UsdPhysics
from modules.short_parallel_gripper import build_short_parallel_gripper
stage=Usd.Stage.Open('/ws/workbench.usd',load=Usd.Stage.LoadNone)
profile=build_short_parallel_gripper(stage)
assert not stage.GetPrimAtPath('/Root/ur5/Robotiq_2F_140_physics_edit')
root='/Root/ur5/ShortParallelGripper'
for name in ['wrist_mount','finger_joint','right_finger_joint']:
    joint=UsdPhysics.Joint(stage.GetPrimAtPath(root+'/'+name))
    anchors=[]
    for relation,attr in [(joint.GetBody0Rel(),joint.GetLocalPos0Attr()),(joint.GetBody1Rel(),joint.GetLocalPos1Attr())]:
        body=stage.GetPrimAtPath(relation.GetTargets()[0])
        matrix=np.asarray(UsdGeom.Xformable(body).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
        anchors.append((matrix@np.r_[list(attr.Get()),1.])[:3])
    assert np.linalg.norm(anchors[0]-anchors[1])<1e-6,(name,anchors)
master=UsdPhysics.PrismaticJoint(stage.GetPrimAtPath(root+'/finger_joint'))
assert abs(master.GetUpperLimitAttr().Get()-.02)<1e-6
assert profile['tcp_offset_m']==.185
print('PASS: replacement authored, mount and slider anchors coincide, limits and TCP validated')
