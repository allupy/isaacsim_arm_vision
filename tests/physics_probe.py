"""Physics-only plant experiment. No ROS, detector, IK, or trajectory converter.

Compare original zero drive targets with preserved authored targets. Commands
are applied by joint name, and all results come from PhysX state feedback.
"""
import os
import json
import time
from pathlib import Path
from isaacsim import SimulationApp
app = SimulationApp({'headless':True, 'limit_cpu_threads':8, 'multi_gpu':False,
                     'create_new_stage':False,'disable_viewport_updates':True})
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics
import numpy as np
from isaacsim.core.api import SimulationContext
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction

mode = os.environ.get('PROBE_MODE','original')
units = float(os.environ.get('PROBE_UNITS','.01'))
output = Path('/experiment/diagnostics')/f'physics_{mode}_{units:g}'
output.mkdir(exist_ok=True)
ctx = omni.usd.get_context()
ctx.open_stage('/ws/workbench.usd',None)
stage = ctx.get_stage()
UsdGeom.SetStageMetersPerUnit(stage,units)
for path in [str(p.GetPath()) for p in stage.Traverse() if p.GetTypeName() == 'OmniGraph']:
    stage.RemovePrim(path)
if stage.GetPrimAtPath('/Render'):
    stage.RemovePrim('/Render')
authored = {}
for prim in stage.Traverse():
    if prim.GetTypeName() == 'PhysicsRevoluteJoint':
        angle = prim.GetAttribute('state:angular:physics:position').Get()
        if angle is not None:
            authored[prim.GetName()] = np.deg2rad(float(angle))
            if mode != 'original':
                target = prim.GetAttribute('drive:angular:physics:targetPosition')
                if target.IsValid():
                    target.Set(float(angle))
physics_path = next(str(p.GetPath()) for p in stage.Traverse() if p.GetTypeName()=='PhysicsScene')
sim = SimulationContext(physics_dt=1/120,rendering_dt=1/60,stage_units_in_meters=units,
                        physics_prim_path=physics_path,set_defaults=False,device='cpu')
sim.initialize_physics()
sim.play()
arm = SingleArticulation('/Root/ur5', name='probe_arm', reset_xform_properties=False)
arm.initialize()
names = list(arm.dof_names)
indices = [names.index(n) for n in ['shoulder_pan_joint','shoulder_lift_joint','elbow_joint','wrist_1_joint','wrist_2_joint','wrist_3_joint']]
controller = arm.get_articulation_controller()
samples = []
def snapshot(phase,step,target=None):
    joints = np.asarray(arm.get_joint_positions()).astype(float)
    matrix = np.array(UsdGeom.Xformable(stage.GetPrimAtPath('/Root/ur5/wrist_3_link')).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
    row = dict(phase=phase,step=step,wall_time=time.time(),sim_time=sim.current_time,
               positions=dict(zip(names,joints.tolist())),wrist=matrix.tolist())
    if target is not None:
        row['error'] = (joints[indices]-target[indices]).tolist()
    samples.append(row)
    if step%120 == 0:
        print(json.dumps(row),flush=True)
    return joints

initial = snapshot('initial',0)
for step in range(1,601):
    sim.step(render=False)
    if step%30 == 0:
        snapshot('settle',step)
settled = np.asarray(arm.get_joint_positions()).astype(float)
target = settled.copy()
target[indices[0]] += .15
controller.apply_action(ArticulationAction(joint_positions=target))
for step in range(1,601):
    sim.step(render=False)
    if step%30 == 0:
        final = snapshot('small_joint_step',step,target)
result = dict(mode=mode,units=units,dof_names=names,authored=authored,
              initial=initial.tolist(),settled=settled.tolist(),target=target.tolist(),final=final.tolist(),
              drift_from_authored_max_rad=float(max(abs(settled[names.index(n)]-q) for n,q in authored.items() if n in names and n in [names[i] for i in indices])),
              final_arm_error_max_rad=float(np.max(np.abs(final[indices]-target[indices]))),
              samples=samples)
(output/'result.json').write_text(json.dumps(result,indent=2))
print('PHYSICS_RESULT '+json.dumps({k:v for k,v in result.items() if k!='samples'}),flush=True)
sim.stop()
app.close()
