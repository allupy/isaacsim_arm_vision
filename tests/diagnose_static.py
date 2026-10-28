"""SDK-only USD inspection; no renderer, simulation, or source-stage writes."""
import glob
import json
from pathlib import Path
import sys
sys.path.insert(0, glob.glob('/isaac-sim/extscache/omni.usd.libs-*')[0])
from pxr import Usd, UsdGeom

stage = Usd.Stage.Open('/ws/workbench.usd', load=Usd.Stage.LoadNone)
result = {'meters_per_unit': UsdGeom.GetStageMetersPerUnit(stage), 'joints':{}, 'links':{}, 'graphs':{}}
for prim in stage.Traverse():
    path = str(prim.GetPath())
    if 'Joint' in prim.GetTypeName():
        result['joints'][prim.GetName()] = {
            'path':path, 'type':prim.GetTypeName(),
            'attributes':{a.GetName(): str(a.Get()) for a in prim.GetAttributes()
                if any(s in a.GetName() for s in ['physics:', 'physx'])},
            'relationships':{r.GetName():[str(p) for p in r.GetTargets()] for r in prim.GetRelationships()}}
    if prim.GetName() in ['base_link','shoulder_link','upper_arm_link','forearm_link','wrist_1_link','wrist_2_link','wrist_3_link']:
        matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        result['links'][prim.GetName()] = {'path':path, 'world_matrix_row_major':[list(row) for row in matrix],
            'attributes':{a.GetName():str(a.Get()) for a in prim.GetAttributes() if 'physics:' in a.GetName()}}
    if 'ROS_JointStates' in path:
        result['graphs'][path] = {a.GetName():{'value':str(a.Get()),'connections':[str(c) for c in a.GetConnections()]}
            for a in prim.GetAttributes() if a.GetName().startswith('inputs:')}
out = Path('/experiment/diagnostics')
out.mkdir(exist_ok=True)
(out/'static_scene.json').write_text(json.dumps(result,indent=2))
print(json.dumps({'meters_per_unit':result['meters_per_unit'],
                  'joint_names':list(result['joints']), 'links':list(result['links'])}))
