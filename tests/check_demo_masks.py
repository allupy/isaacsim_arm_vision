import json,sys
from pathlib import Path
import numpy as np,cv2
sys.path.insert(0,'/experiment/src')
from config.config import DetectorConfig
from modules.detection_manager import DetectionManager
from modules.feedback_observation import mask_geometry,reject_support_pixels
cfg=DetectorConfig.from_yaml('/experiment/src/config/default_config.yaml')
cfg.model_path='/ros2_ws/isaacsim_arm_vision/yolov8s-worldv2.pt'
cfg.sam_checkpoint='/ros2_ws/vision/sam_vit_b_01ec64.pth';cfg.sam_model_type='vit_b'
cfg.confidence_threshold=.02;cfg.default_classes=['cup','banana','apple'];cfg.sam_prompt_mode='box'
det=DetectionManager(cfg)
root=Path('/experiment/runs/yolo_cup_20261028_1')
for prefix in ['001_wrist','002_wrist']:
    rgb=cv2.imread(str(root/(prefix+'.png')));data=np.load(root/(prefix+'.npz'))
    calib=json.loads(str(data['calibration']))
    _,detections=det.detect(rgb,segment_all=True)
    for d in detections:
        mask,plane=reject_support_pixels(data['depth'],d.mask,calib)
        try: geometry=mask_geometry(rgb,data['depth'],mask,calib,True)
        except ValueError as e: geometry=str(e)
        print(prefix,d.class_name,d.confidence,'plane',plane,'geometry',geometry,flush=True)
        image=rgb.copy();image[mask>0]=(image[mask>0]*.4+np.array([0,180,0])*.6).astype('uint8')
        cv2.imwrite('/experiment/diagnostics/'+prefix+'_'+d.class_name+'_boxmask.png',image)
