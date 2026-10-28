"""Host-side recording orchestration; actuator commands remain in the ROS container."""
import argparse
import json
from pathlib import Path
import subprocess
import time

parser=argparse.ArgumentParser()
parser.add_argument('--root',default='/home/aluupy/arm_feedback_experiments/20261027')
parser.add_argument('--prefix',default='record20261028')
parser.add_argument('--count',type=int,default=3)
parser.add_argument('--start-index',type=int,default=1)
parser.add_argument('--tcp-offset',type=float,default=.185)
parser.add_argument('--safe-z',type=float,default=.5)
parser.add_argument('--gripper-profile',choices=['robotiq_2f140','short_parallel'],default='robotiq_2f140')
parser.add_argument('--perception',choices=['color','yolo_sam'],default='color')
parser.add_argument('--instruction')
parser.add_argument('--target-label',default='red')
parser.add_argument('--place-z',type=float,default=.06)
parser.add_argument('--locked-sam-tracking',action='store_true')
parser.add_argument('--det-confidence',type=float,default=.02)
args=parser.parse_args()
root=Path(args.root)

def request(data):
    path=root/'sim/video_request.json'
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(data));temp.replace(path)

def wait_status(identifier,state,timeout=300):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        path=root/'sim/video_state.json'
        if path.exists():
            try:
                status=json.loads(path.read_text())
                if status['id']==identifier and status['state']==state:
                    return status
            except json.JSONDecodeError:
                pass
        time.sleep(.2)
    raise TimeoutError('Recording status unavailable: '+identifier+' '+state)

for i in range(args.start_index,args.start_index+args.count):
    identifier=f'{args.prefix}_{i}'
    output=root/'runs'/identifier
    if output.exists() or (root/'videos'/identifier).exists():
        raise FileExistsError(identifier)
    (root/'sim/reset_scene_requested').write_text(identifier)
    deadline=time.monotonic()+300
    while not ((root/'sim/reset_scene_ack').exists() and (root/'sim/reset_scene_ack').read_text()==identifier):
        if time.monotonic()>deadline:
            raise TimeoutError('Scene reset did not complete')
        time.sleep(.5)
    time.sleep(1.)
    request(dict(action='start',id=identifier))
    wait_status(identifier,'recording')
    command=['docker','exec','arm-feedback-control-20261027','bash','/experiment/src/sim_scene/run_feedback.sh',
             '--output','/experiment/runs/'+identifier,'--perception',args.perception,
             '--home','.45','-.12','.5','--place-surface','.4','-.3',str(args.place_z),
             '--tcp-offset',str(args.tcp_offset),'--expected-gripper',args.gripper_profile,'--safe-z',str(args.safe_z)]
    command += (['--instruction',args.instruction] if args.instruction else ['--target-label',args.target_label])
    if args.locked_sam_tracking:
        command.append('--locked-sam-tracking')
    command+=['--det-confidence',str(args.det_confidence)]
    try:
        with (root/'runs'/f'{identifier}.log').open('w') as log:
            completed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=360)
        print(identifier,'exit',completed.returncode,flush=True)
    finally:
        time.sleep(1.)
        request(dict(action='stop',id=identifier))
        status=wait_status(identifier,'finished',timeout=60)
        print(json.dumps(status),flush=True)
    result=json.loads((output/'result.json').read_text())
    print(json.dumps(dict(run=identifier,success=result.get('success'),elapsed_s=result.get('elapsed_s'),
                          evaluation=result.get('ground_truth_evaluation'))),flush=True)
