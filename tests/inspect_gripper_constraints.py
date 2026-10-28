"""Read-only USD constraint audit, without starting SimulationApp or physics."""
import glob
import json
import sys
sys.path.insert(0,glob.glob('/isaac-sim/extscache/omni.usd.libs-*')[0])
from pxr import Usd
stage=Usd.Stage.Open('/ws/workbench.usd',load=Usd.Stage.LoadNone)
rows=[]
for prim in stage.Traverse():
    if 'Robotiq_2F_140' not in str(prim.GetPath()) or 'Joint' not in prim.GetTypeName():
        continue
    def value(name):
        attr=prim.GetAttribute(name)
        return str(attr.Get()) if attr else None
    rows.append(dict(name=prim.GetName(),type=prim.GetTypeName(),
        limits=[value('physics:lowerLimit'),value('physics:upperLimit')],
        excluded=value('physics:excludeFromArticulation'),
        mimic={r.GetName():[str(p) for p in r.GetTargets()] for r in prim.GetRelationships() if 'mimic' in r.GetName().lower()},
        stiffness=value('drive:angular:physics:stiffness'),damping=value('drive:angular:physics:damping')))
print(json.dumps(rows,indent=2))
