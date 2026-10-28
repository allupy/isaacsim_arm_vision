"""YOLO overlays recomputed on recorded RGB, for display only (not policy evidence)."""
import argparse,json
from pathlib import Path
import cv2
from ultralytics import YOLOWorld
p=argparse.ArgumentParser();p.add_argument('--id',required=True);args=p.parse_args()
root=Path('/experiment');run=root/'runs'/args.id;source=root/'videos'/args.id
result=json.loads((run/'result.json').read_text())
model=YOLOWorld('/ros2_ws/isaacsim_arm_vision/yolov8s-worldv2.pt')
classes=result['detector_classes'];model.set_classes(classes)
output={}
for line in json.loads((source/'frames.json').read_text()):
    index=line['index'];pair=[]
    for camera in ['observer','wrist']:
        frame=cv2.imread(str(source/f'{index:06d}_{camera}.jpg'))
        prediction=model.predict(frame,imgsz=1280,conf=result.get('detector_confidence',.02),max_det=8,verbose=False)[0]
        pair.append([dict(label=classes[int(b.cls)],confidence=float(b.conf),box=b.xyxy.tolist()[0]) for b in prediction.boxes])
    output[str(index)]=pair
(source/'display_detections.json').write_text(json.dumps(output))
print(args.id,'annotated',len(output),'source frame pairs',flush=True)
