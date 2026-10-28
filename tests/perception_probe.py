"""Offline prompt/threshold diagnostic on recorded simulator images.

Colored connected components provide diagnostic image regions, not training
labels and not policy inputs for the open-vocabulary detection path.
"""
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from ultralytics import YOLOWorld
from modules.feedback_observation import color_masks

root = Path('/experiment')
out = root/'diagnostics'/('perception_corrected_view' if os.environ.get('PROBE_IMAGE') else 'perception_prompt_probe')
out.mkdir(exist_ok=True)
prompts = [['cube'],['cube','block'],['box','block','toy'],
           ['red cube','green cube','blue cube'],['red block','green block','blue block']]
images = [root/'runs/sensor_color_02/002_observer.png',root/'runs/sensor_yolo_01/002_observer.png',
          root/'runs/sensor_color_02/001_wrist.png']
if os.environ.get('PROBE_IMAGE'):
    images=[Path(os.environ['PROBE_IMAGE'])]
def iou(a,b):
    low=np.maximum(a[:2],b[:2]); high=np.minimum(a[2:],b[2:])
    inter=np.maximum(0,high-low).prod()
    return float(inter/(np.maximum(0,a[2:]-a[:2]).prod()+np.maximum(0,b[2:]-b[:2]).prod()-inter+1e-9))
rows=[]
for pi,classes in enumerate(prompts):
    # This installed Ultralytics version retains a CPU CLIP encoder after GPU
    # prediction; use a fresh instance for each independently tested prompt set.
    model = YOLOWorld('/ros2_ws/isaacsim_arm_vision/yolov8s-worldv2.pt')
    model.set_classes(classes)
    for image_path in images:
        rgb=cv2.imread(str(image_path))
        if rgb is None:
            continue
        regions=[]
        for color,mask,area in color_masks(rgb):
            ys,xs=np.where(mask)
            regions.append((color,[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)]))
        result=model.predict(rgb,conf=.001,iou=.5,imgsz=1280,max_det=50,verbose=False)[0]
        boxes=[dict(label=result.names[int(b.cls[0])],score=float(b.conf[0]),bbox=b.xyxy[0].cpu().numpy().tolist()) for b in result.boxes]
        matches=[]
        for color,region in regions:
            overlapping=[b for b in boxes if iou(np.asarray(b['bbox']),np.asarray(region))>.3]
            best=max(overlapping,key=lambda b:b['score']) if overlapping else None
            matches.append(dict(color=color,region=region,best=best))
        row=dict(prompt=classes,image=str(image_path),boxes=boxes,matches=matches)
        rows.append(row)
        cv2.imwrite(str(out/f'p{pi}_{image_path.parent.name}_{image_path.stem}.png'),result.plot())
        print(json.dumps(dict(prompt=classes,image=str(image_path),matches=matches)),flush=True)
    (out/'result.json').write_text(json.dumps(rows,indent=2))
(out/'result.json').write_text(json.dumps(rows,indent=2))
