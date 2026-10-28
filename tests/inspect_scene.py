import glob
import sys
sys.path.insert(0, glob.glob('/isaac-sim/extscache/omni.usd.libs-*')[0])
from pxr import Sdf, Usd
layer = Sdf.Layer.FindOrOpen('/ws/workbench.usd')
print('REFERENCES',layer.GetExternalReferences(),flush=True)
stage = Usd.Stage.Open(layer, load=Usd.Stage.LoadNone)
for prim in stage.Traverse():
    path = str(prim.GetPath())
    if 'Joint' in prim.GetTypeName() or prim.GetName() in ['articulation_controller','ArticulationController']:
        print('PRIM',path,prim.GetTypeName())
        for attr in prim.GetAttributes():
            if any(key in attr.GetName().lower() for key in ['drive','axis','limit','joint','position','stiffness','damping']):
                print(' ',attr.GetName(),attr.Get())
    if 'ROS_JointStates' in path:
        print('GRAPH',path,[(attr.GetName(),str(attr.Get())) for attr in prim.GetAttributes() if attr.GetName().startswith('inputs:')])
