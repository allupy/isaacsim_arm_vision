"""Measured empty close/open check before attempting object grasping."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from types import SimpleNamespace
import json
import rclpy
from feedback_runner import SensorBackend
from modules.feedback_motion import FeedbackArmController

output=Path(sys.argv[1]);output.mkdir(exist_ok=False,parents=True)
rclpy.init()
args=SimpleNamespace(perception='color',sensor_timeout=30.,config='config/default_config.yaml')
backend=SensorBackend(args,output)
try:
    if not backend.wait_ready(240.):raise TimeoutError('sensor readiness')
    if backend.latest_calibration['gripper']['id']!='short_parallel':raise ValueError('wrong gripper')
    backend.controller=FeedbackArmController(backend)
    backend.pump(.4)
    records=[]
    for closed in [False,True,False]:
        ok=backend.grip(closed)
        backend.pump(.5)
        js=backend.controller.latest_joint_state
        q=dict(zip(js.name,js.position))
        master=q['finger_joint']*backend.latest_calibration['gripper']['command_scale']
        follower=q['right_finger_joint']
        records.append(dict(closed=closed,success=ok,master_m=master,follower_m=follower,
                            coupling_error_m=abs(master+follower)))
    result=dict(success=all(r['success'] and r['coupling_error_m']<.001 for r in records),records=records)
    (output/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
    if not result['success']:raise RuntimeError('Measured open/close validation failed')
finally:
    backend.hold();backend.writer.shutdown(wait=True);backend.evaluator.file.close()
    if backend.controller:backend.controller.shutdown()
    backend.destroy_node();rclpy.shutdown()
