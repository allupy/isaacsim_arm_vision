import json
from pathlib import Path
from ultralytics import YOLOWorld
model=YOLOWorld('/ros2_ws/isaacsim_arm_vision/yolov8s-worldv2.pt')
prompts=['red cube','red block','red object','cube','block','toy block','red toy','green cube','blue cube','cup','mug','banana','apple']
model.set_classes(prompts)
root=Path('/experiment/runs/yolo_cube_red_20261028_1')
for name in ['001_wrist.png','002_wrist.png','003_wrist.png','004_observer.png']:
    result=model.predict(str(root/name),imgsz=1280,conf=.001,max_det=30,verbose=False)[0]
    print(name,json.dumps([(prompts[int(b.cls)],round(float(b.conf),4),b.xyxy.tolist()[0]) for b in result.boxes]),flush=True)
