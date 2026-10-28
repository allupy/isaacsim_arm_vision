import sys,json
from pathlib import Path
sys.path.insert(0,'/experiment/src')
from feedback_runner import EvaluationRecorder
root=Path('/experiment/runs')
for folder in sorted(root.glob('yolo_*20261028_1')):
    if not (folder/'result.json').exists():continue
    result=json.loads((folder/'result.json').read_text())
    records=[json.loads(line) for line in (folder/'ground_truth.jsonl').read_text().splitlines()]
    locked=next((e['locked_target']['position'] for e in result.get('events',[]) if 'locked_target' in e),None)
    if not records or locked is None:continue
    scorer=EvaluationRecorder.__new__(EvaluationRecorder)
    scorer.first=records[0];scorer.last=records[-1]
    scorer.max_height={name:max(r['objects'][name][2] for r in records) for name in records[0]['objects']}
    result['ground_truth_evaluation']=scorer.summary(locked,result['config']['place_surface'],(result.get('command') or {}).get('target'))
    result['evaluation_audit']='Requested identity enforced; original result.json preserved'
    (folder/'evaluation_audit.json').write_text(json.dumps(result,indent=2))
    print(folder.name,result['success'],result['ground_truth_evaluation']['task_success'],flush=True)
