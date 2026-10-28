import glob,sys,json
sys.path.insert(0,glob.glob('/isaac-sim/extscache/omni.usd.libs-*')[0])
from pxr import Usd,UsdGeom
import numpy as np
stage=Usd.Stage.Open('/ws/workbench.usd',load=Usd.Stage.LoadNone)
wrist=np.array(UsdGeom.Xformable(stage.GetPrimAtPath('/Root/ur5/wrist_3_link')).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
inverse=np.linalg.inv(wrist)
cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(),[UsdGeom.Tokens.default_])
rows=[]
for prim in stage.Traverse():
    if prim.GetName() in ['left_inner_finger','right_inner_finger','left_inner_knuckle','right_inner_knuckle']:
        matrix=np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
        box=cache.ComputeWorldBound(prim).ComputeAlignedBox()
        low=np.array(box.GetMin()); high=np.array(box.GetMax())
        points=np.array([inverse @ np.r_[ [x,y,z],1.] for x in [low[0],high[0]] for y in [low[1],high[1]] for z in [low[2],high[2]]])
        rows.append(dict(path=str(prim.GetPath()),center_in_wrist=(inverse@matrix)[:3,3].tolist(),
                         bounds_wrist_low=points[:,:3].min(axis=0).tolist(),bounds_wrist_high=points[:,:3].max(axis=0).tolist()))
print(json.dumps(rows,indent=2))
open('/experiment/diagnostics/gripper_geometry.json','w').write(json.dumps(rows,indent=2))
