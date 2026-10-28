"""One measured home motion, evaluated against independent simulator pose."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from types import SimpleNamespace
import json
import numpy as np
import rclpy
from scipy.spatial.transform import Rotation
from feedback_runner import SensorBackend
from modules.feedback_motion import FeedbackArmController

output = Path(sys.argv[1])
output.mkdir(exist_ok=False,parents=True)
rclpy.init()
args=SimpleNamespace(perception='color',sensor_timeout=30.,config='config/default_config.yaml')
backend=SensorBackend(args,output)
try:
    if not backend.wait_ready(180.):
        raise TimeoutError('startup_sensor_not_ready')
    backend.controller=FeedbackArmController(backend)
    backend.observe('observer')
    goal=[.45,-.12,.5]
    ok=backend.move(goal,[1.,0.,0.,0.])
    backend.observe('observer')
    truth=backend.evaluator.last
    result={'motion_success':ok,'goal':goal,'motion':backend.controller.last_motion}
    if truth:
        actual=np.array(truth['wrist'])
        result['actual_wrist']=actual.tolist()
        result['position_error_m']=float(np.linalg.norm(actual[:3,3]-goal))
        result['rotation_error_deg']=float(np.rad2deg((Rotation.from_quat([1,0,0,0]).inv()*Rotation.from_matrix(actual[:3,:3])).magnitude()))
    (output/'result.json').write_text(json.dumps(result,indent=2))
    print('HOME_PROBE_RESULT '+json.dumps(result),flush=True)
finally:
    backend.hold()
    backend.evaluator.file.close()
    if backend.controller:
        backend.controller.shutdown()
    backend.destroy_node()
    rclpy.shutdown()
