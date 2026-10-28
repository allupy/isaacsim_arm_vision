"""YOLO-World bbox and optional SAM timing on the same fixed RGB requests as LA."""
import argparse
import json
from pathlib import Path
import sys
import time
import hashlib
import cv2
import numpy as np
import torch
from ultralytics import YOLOWorld

parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);args=parser.parse_args()
out=Path(args.out);model_out=out/'yolo_world';model_out.mkdir(exist_ok=False)
cases=json.loads((out/'manifest.json').read_text());classes=['red cube','green cube','blue cube','yellow cube']
def read_image(path):
    mapped=Path(path.replace('/home/aluupy/arm_feedback_experiments/20261027','/experiment'))
    image=cv2.imread(str(mapped))
    if image is None:raise FileNotFoundError(mapped)
    case=next(c for c in cases if c['image']==path)
    if hashlib.sha256(mapped.read_bytes()).hexdigest()!=case['rgb_sha256']:
        raise ValueError('Input checksum mismatch')
    return image
torch.set_num_threads(6)
started=time.monotonic()
model=YOLOWorld('/ros2_ws/isaacsim_arm_vision/yolov8s-worldv2.pt')
model.set_classes(classes)
setup_s=time.monotonic()-started

def iou(a,b):
    low=np.maximum(a[:2],b[:2]);high=np.minimum(a[2:],b[2:]);inter=np.maximum(0.,high-low).prod()
    return float(inter/max(1e-12,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter))

def predict(image,size):
    return model.predict(image,imgsz=size,conf=.001,iou=.5,max_det=50,device=0,verbose=False)[0]

frames={c['frame']:c['image'] for c in cases};cache={};timings={}
for size in [640,1280]:
    first=read_image(next(iter(frames.values())))
    for _ in range(3):predict(first,size)
    for frame,path in frames.items():
        image=read_image(path);samples=[]
        for _ in range(5):
            torch.cuda.synchronize();start=time.perf_counter();result=predict(image,size)
            torch.cuda.synchronize();samples.append(time.perf_counter()-start)
        boxes=[dict(label=result.names[int(b.cls[0])],confidence=float(b.conf[0]),
                    bbox=b.xyxy[0].cpu().tolist()) for b in result.boxes]
        cache[(size,frame)]=boxes;timings[(size,frame)]=samples
        cv2.imwrite(str(model_out/f'{frame}_{size}.jpg'),result.plot())
rows=[]
for size in [640,1280]:
    for threshold in [.02,.25]:
        for case in cases:
            color=case['target'] or 'yellow';label=color+' cube'
            candidates=[b for b in cache[(size,case['frame'])] if b['label']==label and b['confidence']>=threshold]
            chosen=max(candidates,key=lambda b:b['confidence']) if candidates else None
            overlap=None if case['target'] is None else (iou(chosen['bbox'],case['reference'][color]['box']) if chosen else 0.)
            rows.append(dict(id=case['id'],frame=case['frame'],target=case['target'],imgsz=size,conf=threshold,
                box=None if chosen is None else chosen['bbox'],score=None if chosen is None else chosen['confidence'],
                localized=bool(case['target'] and overlap>=.5),target_iou=overlap,
                correct_absence=case['target'] is None and chosen is None,
                median_frame_latency_s=float(np.median(timings[(size,case['frame'])]))))
metadata=dict(weights='yolov8s-worldv2.pt',text_setup_and_load_s=setup_s,
    torch=torch.__version__,gpu=torch.cuda.get_device_name(0),vocabulary=classes,
    note='All four text embeddings cached before warm frame timing; highest-scoring box of requested class')
(model_out/'bbox_results.json').write_text(json.dumps(dict(metadata=metadata,rows=rows,
    timings=[dict(imgsz=s,frame=f,samples=v) for (s,f),v in timings.items()],
    raw_predictions=[dict(imgsz=s,frame=f,boxes=v) for (s,f),v in cache.items()]),indent=2))
print('BBOX_DONE',json.dumps(metadata),flush=True)

# An additional pipeline measurement: 1280 YOLO at conf .02 + SAM ViT-B for one
# selected box. The full SAM image encoder is included on every timed request.
from segment_anything import sam_model_registry,SamPredictor
start=time.monotonic()
sam=SamPredictor(sam_model_registry['vit_b'](checkpoint='/ros2_ws/vision/sam_vit_b_01ec64.pth').to('cuda').eval())
sam_load_s=time.monotonic()-start
first=read_image(next(iter(frames.values())));sam.set_image(cv2.cvtColor(first,cv2.COLOR_BGR2RGB))
sam.predict(box=np.array([0,0,30,30]),multimask_output=False)
sam_rows=[]
for row in rows:
    if row['imgsz']!=1280 or row['conf']!=.02 or row['box'] is None:continue
    case=next(c for c in cases if c['id']==row['id']);image=read_image(case['image'])
    durations=[]
    for _ in range(3):
        torch.cuda.synchronize();start=time.perf_counter()
        sam.set_image(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        masks,scores,_=sam.predict(box=np.asarray(row['box']),multimask_output=False)
        torch.cuda.synchronize();durations.append(time.perf_counter()-start)
    overlap=None
    if case['target']:
        reference=np.load(out/f'{case["frame"]}_{case["target"]}_reference.npy')
        mask=masks[0]>0
        overlap=float((mask&reference).sum()/max(1,(mask|reference).sum()))
        np.save(model_out/f'{case["id"]}_sam.npy',mask)
    sam_rows.append(dict(id=case['id'],target=case['target'],sam_median_s=float(np.median(durations)),
        estimated_yolo_sam_s=row['median_frame_latency_s']+float(np.median(durations)),mask_iou=overlap))
(model_out/'sam_results.json').write_text(json.dumps(dict(sam_load_s=sam_load_s,
    mode='SAM ViT-B bbox prompt only; no HSV/GT box or point prompt',rows=sam_rows),indent=2))
print('SAM_DONE',len(sam_rows),flush=True)
