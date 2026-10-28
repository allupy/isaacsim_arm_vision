"""GPU-synchronized stage timing through the existing RGB-D grasp geometry.

Offline: recorded RGB-D pairs. Live: read-only ROS observation, no arm controller.
The result is a visible surface point plus fixed downward grasp orientation,
not general six-degree-of-freedom object-pose registration.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from ultralytics import YOLOWorld
from segment_anything import sam_model_registry,SamPredictor
sys.path.insert(0,'/experiment/src')
from modules.feedback_observation import mask_geometry

class Pipeline:
    def __init__(self):
        torch.set_num_threads(6)
        self.detector=YOLOWorld('/ros2_ws/isaacsim_arm_vision/yolov8s-worldv2.pt')
        self.detector.set_classes(['red cube','green cube','blue cube','yellow cube'])
        self.sam=SamPredictor(sam_model_registry['vit_b'](checkpoint='/ros2_ws/vision/sam_vit_b_01ec64.pth').to('cuda').eval())

    @torch.inference_mode()
    def process(self,rgb,depth,calibration,target,planning_to_world=None):
        timings={};torch.cuda.synchronize();started=time.perf_counter();phase=started
        result=self.detector.predict(rgb,imgsz=1280,conf=.02,iou=.5,max_det=50,device=0,verbose=False)[0]
        torch.cuda.synchronize();now=time.perf_counter();timings['yolo_s']=now-phase;phase=now
        matches=[b for b in result.boxes if result.names[int(b.cls[0])]==target]
        if not matches:
            return dict(success=False,reason='target_not_detected',target=target,
                        compute_s=time.perf_counter()-started,timings=timings)
        chosen=max(matches,key=lambda b:float(b.conf[0]));box=chosen.xyxy[0].cpu().numpy()
        timings['selection_s']=time.perf_counter()-phase;phase=time.perf_counter()
        self.sam.set_image(cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB))
        torch.cuda.synchronize();now=time.perf_counter();timings['sam_encoder_s']=now-phase;phase=now
        masks,_,_=self.sam.predict(box=box,multimask_output=False)
        torch.cuda.synchronize();now=time.perf_counter();timings['sam_decoder_s']=now-phase;phase=now
        try:
            position,appearance,_,extent=mask_geometry(rgb,depth,masks[0],calibration,return_details=True)
        except ValueError as exc:
            return dict(success=False,reason='geometry:'+str(exc),target=target,
                        compute_s=time.perf_counter()-started,timings=timings)
        now=time.perf_counter();timings['mask_geometry_s']=now-phase;phase=now
        quaternion=[1.,0.,0.,0.]
        # The currently deployed cube policy prescribes downward orientation.
        # TCP offset is the original/adapter gripper's current 185 mm setting.
        grasp_world=np.asarray(position)+[0.,0.,.185]
        base=np.eye(4) if planning_to_world is None else np.asarray(planning_to_world)
        inverse=np.linalg.inv(base)
        grasp_planning=(inverse@np.r_[grasp_world,1.])[:3].tolist()
        local_q=(Rotation.from_matrix(inverse[:3,:3])*Rotation.from_quat(quaternion)).as_quat().tolist()
        timings['grasp_transform_s']=time.perf_counter()-phase
        return dict(success=True,target=target,box=box.tolist(),confidence=float(chosen.conf[0]),
            surface_world=position,grasp_world=grasp_world.tolist(),grasp_planning=grasp_planning,
            orientation_planning=local_q,extent_xy=extent,compute_s=time.perf_counter()-started,timings=timings,
            orientation_mode='fixed downward; not estimated arbitrary object orientation')

def read_pair(case):
    path=Path(case['image'].replace('/home/aluupy/arm_feedback_experiments/20261027','/experiment'))
    if hashlib.sha256(path.read_bytes()).hexdigest()!=case['rgb_sha256']:raise ValueError('RGB checksum mismatch')
    rgb=cv2.imread(str(path));packet=np.load(path.with_suffix('.npz'))
    depth=packet['depth'];calibration=json.loads(str(packet['calibration']))
    if rgb is None or rgb.shape[:2]!=depth.shape:raise ValueError('RGB-D shape mismatch')
    return rgb,depth,calibration

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['offline','live'],required=True)
    parser.add_argument('--out',required=True);parser.add_argument('--count',type=int,default=12)
    args=parser.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if (out/f'{args.mode}.json').exists():raise FileExistsError('Existing result')
    start=time.perf_counter();pipeline=Pipeline();load_s=time.perf_counter()-start
    cases=json.loads(Path('/experiment/diagnostics/yolo_la_compare_20261028/manifest.json').read_text())
    first=next(c for c in cases if c['target']=='red')
    inputs=read_pair(first)
    for _ in range(3):pipeline.process(*inputs,'red cube')
    rows=[]
    if args.mode=='offline':
        for case in cases:
            rgb,depth,calibration=read_pair(case)
            target=(case['target'] or 'yellow')+' cube'
            for repeat in range(3):
                row=pipeline.process(rgb,depth,calibration,target)
                rows.append(dict(id=case['id'],repeat=repeat,**row))
        result=dict(mode='recorded RGB-D; I/O excluded',load_s=load_s,rows=rows)
    else:
        import rclpy
        from std_msgs.msg import String
        from types import SimpleNamespace
        from feedback_runner import SensorBackend
        rclpy.init();node=SensorBackend(SimpleNamespace(perception='color',sensor_timeout=25.,
                                    config='config/default_config.yaml'),out)
        try:
            if not node.wait_ready(300.):raise TimeoutError('No live camera')
            last_stamp=0
            for i in range(args.count):
                start=time.perf_counter();barrier=max(last_stamp,node.get_clock().now().nanoseconds)
                node.camera_request_pub.publish(String(data='observer'))
                deadline=time.monotonic()+25;packet=None
                while time.monotonic()<deadline:
                    node.pump(.01);packet=node.buffers['observer'].latest(after_stamp=barrier)
                    if packet:break
                if packet is None:raise TimeoutError('Live fresh RGB-D timeout')
                wait_s=time.perf_counter()-start;last_stamp,data=packet
                received_age_s=(node.get_clock().now().nanoseconds-last_stamp)/1e9
                row=pipeline.process(data['rgb'],data['depth'],data['calibration'],'green cube',
                                     node.latest_calibration['planning_to_world'])
                row.update(sample=i,wait_fresh_rgbd_s=wait_s,frame_age_at_ready_s=received_age_s,
                           request_to_pose_s=time.perf_counter()-start,stamp_ns=last_stamp)
                rows.append(row);print(json.dumps(row),flush=True)
            result=dict(mode='live fixed observer camera; green cube; no arm motion',load_s=load_s,rows=rows)
        finally:
            node.writer.shutdown(wait=True);node.evaluator.file.close();node.destroy_node();rclpy.shutdown()
    (out/f'{args.mode}.json').write_text(json.dumps(result,indent=2))
    print('DONE',args.mode,'successes',sum(r['success'] for r in rows),'/',len(rows),flush=True)

if __name__=='__main__':main()
