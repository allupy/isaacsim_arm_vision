import json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np,cv2
sys.path.insert(0,'/experiment/src')
from segment_anything import sam_model_registry,SamPredictor
from feedback_runner import SensorBackend
from modules.feedback_observation import mask_geometry,reject_support_pixels
root=Path('/experiment/runs/yolo_banana_track2_20261028_1')
result=json.loads((root/'result.json').read_text())
locked=next(e['locked_target'] for e in result['events'] if 'locked_target' in e)
expected=next(e['expected'] for e in result['events'] if e['state']=='VERIFY_LIFT')
sam=sam_model_registry['vit_b'](checkpoint='/ros2_ws/vision/sam_vit_b_01ec64.pth').to('cuda')
backend=SimpleNamespace(detector=SimpleNamespace(sam_predictor=SamPredictor(sam)),log=print)
for prefix in ['003_observer','004_observer']:
    data=np.load(root/(prefix+'.npz'));rgb=cv2.imread(str(root/(prefix+'.png')))
    calib=json.loads(str(data['calibration']))
    mask=SensorBackend.segment_locked_region(backend,rgb,data['depth'],calib,expected,SimpleNamespace(**locked))
    if mask is None:print(prefix,'missing');continue
    mask,_=reject_support_pixels(data['depth'],mask,calib)
    print(prefix,'pixels',int(mask.sum()),'geometry',mask_geometry(rgb,data['depth'],mask,calib,True),flush=True)
    cv2.imwrite('/experiment/diagnostics/'+prefix+'_tracked.png',mask*255)
