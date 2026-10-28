"""Offline timings and independent scoring, never an input to the policy."""
import json
from pathlib import Path
import sys
import numpy as np

root=Path(sys.argv[1])
summaries=[]
for directory in sorted(root.glob('performance_*')):
    if not directory.is_dir() or not (directory/'result.json').exists():
        continue
    result=json.loads((directory/'result.json').read_text())
    events=[json.loads(s) for s in (directory/'events.jsonl').read_text().splitlines()] if (directory/'events.jsonl').exists() else []
    stages={}
    state_events=[e for e in events if 'state' in e]
    for a,b in zip(state_events,state_events[1:]):
        stages[a['state']]=stages.get(a['state'],0.)+b['time']-a['time']
    obs=[e for e in events if e.get('kind')=='observation']
    truth=result.get('ground_truth_evaluation',{})
    summary=dict(run=directory.name,success=result.get('success'),reason=result.get('reason'),
        elapsed_s=result.get('elapsed_s'),truth_success=truth.get('task_success'),
        placement_error_m=truth.get('placement_xy_error'),stages=stages,
        observation_wait_s=[e.get('wait_wall_s') for e in obs],
        observation_processing_s=[e.get('processing_wall_s') for e in obs])
    if state_events and 'sim_time' in state_events[0] and 'sim_time' in state_events[-1]:
        summary['elapsed_sim_s']=state_events[-1]['sim_time']-state_events[0]['sim_time']
    trace=directory/'command_updates.jsonl'
    if trace.exists():
        commands=[json.loads(s) for s in trace.read_text().splitlines()]
        dt=[]; dq=[]
        for a,b in zip(commands,commands[1:]):
            if a[2]==b[2] and 0 < b[0]-a[0] < 2.:
                arm_indices=[i for i,n in enumerate(a[2]) if n in
                    ['shoulder_pan_joint','shoulder_lift_joint','elbow_joint','wrist_1_joint','wrist_2_joint','wrist_3_joint']]
                if not arm_indices:
                    continue
                step=float(np.max(np.abs(np.asarray(b[3])[arm_indices]-np.asarray(a[3])[arm_indices])))
                if step>1e-7:
                    dt.append(b[0]-a[0]);dq.append(step)
        if dt:
            summary['moving_command_gap_sim_s_p95']=float(np.percentile(dt,95))
            summary['moving_command_max_step_rad']=max(dq)
    execution=directory/'execution.jsonl'
    if execution.exists():
        errors=[]
        for line in execution.read_text().splitlines():
            trajectory=json.loads(line)
            indices=[i for i,n in enumerate(trajectory['names']) if n!='finger_joint']
            for sample in trajectory['samples']:
                errors.extend((np.asarray(sample[2])[indices]-np.asarray(sample[3])[indices]).tolist())
        if errors:
            summary['tracking_rms_rad']=float(np.sqrt(np.mean(np.square(errors))))
            summary['tracking_max_rad']=float(np.max(np.abs(errors)))
    summaries.append(summary)
out=root/'performance_summary.json'
out.write_text(json.dumps(summaries,indent=2))
print(json.dumps(summaries,indent=2))
