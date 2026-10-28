"""Workstation test driver: sequential skills without scene reset, then displacement."""
import json,subprocess,time,argparse
from pathlib import Path
root=Path('/home/aluupy/arm_feedback_experiments/20261027')
parser=argparse.ArgumentParser();parser.add_argument('--sequence-only',action='store_true');parser.add_argument('--prefix',default='interactive_sequence_20261028');args=parser.parse_args()
def wait_file(path,value,timeout=300):
    deadline=time.monotonic()+timeout
    while not (path.exists() and path.read_text()==value):
        if time.monotonic()>deadline:raise TimeoutError(str(path))
        time.sleep(.2)
def video(identifier,action):
    path=root/'sim/video_request.json';temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(dict(id=identifier,action=action)));temp.replace(path)
    deadline=time.monotonic()+60
    while time.monotonic()<deadline:
        state=json.loads((root/'sim/video_state.json').read_text()) if (root/'sim/video_state.json').exists() else {}
        if state.get('id')==identifier and state.get('state')==('recording' if action=='start' else 'finished'):return
        time.sleep(.2)
    raise TimeoutError('video acknowledgement')
def reset(identifier):
    (root/'sim/reset_scene_requested').write_text(identifier)
    wait_file(root/'sim/reset_scene_ack',identifier)
    time.sleep(1)
def run(prefix,step,target,xy,perturb=False):
    identifier=f'{prefix}_{step}';output=root/'runs'/identifier
    if output.exists():raise FileExistsError(identifier)
    video(identifier,'start')
    command=['docker','exec','arm-feedback-control-20261027','bash','/experiment/src/sim_scene/run_feedback.sh',
        '--output','/experiment/runs/'+identifier,'--perception','yolo_sam','--instruction',f'Pick the {target}',
        '--home','.45','-.12','.5','--place-surface',str(xy[0]),str(xy[1]),'.06',
        '--tcp-offset','.185','--expected-gripper','short_parallel','--locked-sam-tracking']
    if perturb:command+=['--test-perturb']
    if target=='blue cube':command+=['--acquire-camera','observer']
    try:
        with (root/'runs'/f'{identifier}.log').open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=240)
    finally:
        time.sleep(1);video(identifier,'stop')
    result=json.loads((output/'result.json').read_text())
    print(identifier,json.dumps({k:result.get(k) for k in ['success','reason','ground_truth_evaluation']}),flush=True)
    return result['success'] and result.get('ground_truth_evaluation',{}).get('task_success')
reset(args.prefix)
ok=run(args.prefix,1,'red cube',(.35,-.30))
if ok:run(args.prefix,2,'blue cube',(.49,-.30))
if not args.sequence_only:
    reset('interactive_displace_20261028')
    run('interactive_displace_20261028',1,'red cube',(.4,-.3),True)
