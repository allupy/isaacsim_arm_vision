"""Offline scoring of matched grasp trials; ground truth never enters the policy."""
import json
from pathlib import Path
import sys
import numpy as np
root=Path(sys.argv[1])
rows=[]
for profile,prefix in [('original','gripper_long_20261028'),('direct_short_v1','gripper_short_20261028'),
                       ('adapter_gap80_v2','gripper_short_v2_20261028'),
                       ('short_parallel','gripper_short_v3_20261028')]:
    for i in range(1,4):
        directory=root/'runs'/f'{prefix}_{i}'
        if not (directory/'result.json').exists():continue
        result=json.loads((directory/'result.json').read_text())
        events=[json.loads(s) for s in (directory/'events.jsonl').read_text().splitlines()]
        truth=[json.loads(s) for s in (directory/'ground_truth.jsonl').read_text().splitlines()]
        states=[e for e in events if 'state' in e]
        starts=[e['time'] for e in states if e['state']=='TRANSFER']
        ends=[e['time'] for e in states if e['state']=='PLACE']
        relative=[]
        if starts and ends:
            for sample in truth:
                if starts[0]<=sample['stamp_ns']/1e9<=ends[0]:
                    wrist=np.asarray(sample['wrist'])
                    point=np.asarray(sample['objects']['red_cube'])
                    relative.append(wrist[:3,:3].T@(point-wrist[:3,3]))
        drift={}
        if relative:
            relative=np.asarray(relative)
            deviations=np.linalg.norm(relative-np.median(relative,axis=0),axis=1)*1000
            drift=dict(transport_samples=len(relative),transport_rms_mm=float(np.sqrt(np.mean(deviations**2))),
                       transport_max_mm=float(np.max(deviations)))
        evaluation=result.get('ground_truth_evaluation',{})
        blue_initial=np.asarray(truth[0]['objects']['blue_cube'])
        blue_shift=max(float(np.linalg.norm(np.asarray(s['objects']['blue_cube'])-blue_initial))*1000 for s in truth)
        rows.append(dict(profile=profile,run=directory.name,policy_success=result.get('success'),
                         reason=result.get('reason'),independent_success=evaluation.get('task_success'),
                         final_xy_error_mm=None if not evaluation else evaluation.get('placement_xy_error',0)*1000,
                         elapsed_wall_s=result.get('elapsed_s'),blue_max_displacement_mm=blue_shift,**drift))
(root/'gripper_comparison.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows,indent=2))
