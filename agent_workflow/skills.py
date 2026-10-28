"""Docker/ROS adapter. The agent never issues joint or Cartesian commands directly."""
import json,subprocess,os
from planner import ZONES,HEIGHTS

class GraspSkills:
    def __init__(self,root,run_id):
        self.root,self.run_id=root,run_id
        self.container=os.environ.get('CONTROL_NAME','arm-feedback-control-20261027')
    def pick_and_place(self,step,index):
        identifier=f'{self.run_id}_{index}'
        x,y=ZONES[step.destination] or (0.,0.)
        command=['docker','exec',self.container,'bash','/experiment/src/sim_scene/run_feedback.sh',
          '--output','/experiment/runs/'+identifier,'--perception','yolo_sam','--instruction','Pick the '+step.target,
          '--home','.45','-.12','.5','--place-surface',str(x),str(y),str(HEIGHTS[step.target]),
          '--tcp-offset','.185','--expected-gripper','short_parallel','--locked-sam-tracking']
        if step.destination in ['plate','cup_stand']:
            command+=['--destination-tag',str({'plate':10,'cup_stand':20}[step.destination])]
        with (self.root/'runs'/f'{identifier}.log').open('w') as log:
            try:done=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=240)
            except subprocess.TimeoutExpired:
                # Docker exec may remain alive. Stop the dedicated experiment control
                # container, mark state unknown, and prevent downstream commands.
                subprocess.run(['docker','stop','-t','3',self.container],timeout=15)
                return dict(success=False,reason='skill_timeout',state_known=False)
        file=self.root/'runs'/identifier/'result.json'
        if not file.exists():return dict(success=False,reason='missing_skill_result',exit_code=done.returncode)
        result=json.loads(file.read_text())
        return dict(success=done.returncode==0 and result.get('success') is True,
                    reason=result.get('reason',''),result_path=str(file),target=step.target,destination=step.destination,
                    lift_verified=result.get('lift_verified'),elapsed_s=result.get('elapsed_s'))
