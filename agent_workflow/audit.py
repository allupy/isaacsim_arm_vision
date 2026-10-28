"""Post-run simulation scoring, never an input to planner or skill execution."""
import argparse,json,math
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',default='/home/aluupy/arm_feedback_experiments/20261027');p.add_argument('--run-id',default='agent_sort_20261028');args=p.parse_args()
root=Path(args.root);folder=root/'agent_runs'/args.run_id
agent=json.loads((folder/'result.json').read_text());scores=[];truth=[]
for index in range(1,len(agent['results'])+1):
    run=root/'runs'/f'{args.run_id}_{index}'
    result=json.loads((run/'result.json').read_text());scores.append(result.get('ground_truth_evaluation',{}))
    lines=(run/'ground_truth.jsonl').read_text().splitlines()
    truth.append((json.loads(lines[0]),json.loads(lines[-1])))
continuity=None
if len(truth)==2:
    before=truth[0][1]['objects']['apple'];after=truth[1][0]['objects']['apple']
    continuity=math.dist(before,after)
audit=dict(agent_success=agent['success'],steps=scores,apple_motion_between_steps_m=continuity,
           verified_success=agent['success'] and len(scores)==2 and all(s.get('task_success') for s in scores)
                            and continuity is not None and continuity<.002)
(folder/'audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit),flush=True)
