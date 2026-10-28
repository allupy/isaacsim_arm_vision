"""Measure optical-camera centering and mask completeness, without grasping."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from types import SimpleNamespace
from dataclasses import asdict
import json
import numpy as np
import cv2
import rclpy
from feedback_runner import SensorBackend
from modules.feedback_motion import FeedbackArmController
from modules.feedback_policy import TargetLock,PolicyConfig

out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
args=SimpleNamespace(perception='color',sensor_timeout=25.,config='config/default_config.yaml')
rclpy.init();backend=SensorBackend(args,out)
try:
    if not backend.wait_ready(180.):
        raise TimeoutError('startup_sensor_not_ready')
    backend.controller=FeedbackArmController(backend)
    backend.observe('observer')
    if not backend.move([.45,-.12,.5],[1.,0.,0.,0.]) or not backend.grip(False):
        raise RuntimeError('home_or_gripper_failed')
    initial=backend.observe('wrist')
    if not initial:
        raise RuntimeError('no_initial_target')
    target=max(initial,key=lambda obs:obs.confidence)
    lock=TargetLock(target,PolicyConfig())
    view_pos,view_quat=backend.observation_pose(target.position,.38)
    if not backend.move(view_pos,view_quat):
        raise RuntimeError('view_motion_failed')
    refined=lock.match(backend.observe('wrist'),target.position)
    packet=backend.buffers['wrist'].frames[refined.stamp_ns]
    camera=packet['calibration']
    T=np.array(camera['optical_to_world']);K=np.array(camera['K'])
    def project(point):
        local=np.linalg.inv(T)@np.r_[point,1.]
        pixel=K@local[:3]
        return pixel[:2]/pixel[2]
    principal=K[:2,2]
    coarse_pixel=project(target.position)
    refined_pixel=project(refined.position)
    area_ratio=float(np.prod(refined.extent_xy)/np.prod(target.extent_xy))
    result=dict(mode='camera_centered_observation_only',target=asdict(target),refined=asdict(refined),
                commanded_ee_position=view_pos,commanded_ee_quaternion=view_quat,
                measured_camera_position=T[:3,3].tolist(),principal_point=principal.tolist(),
                coarse_target_projection=coarse_pixel.tolist(),observed_target_projection=refined_pixel.tolist(),
                coarse_target_pixel_error=float(np.linalg.norm(coarse_pixel-principal)),
                observed_target_pixel_error=float(np.linalg.norm(refined_pixel-principal)),
                camera_xy_error_to_coarse_m=float(np.linalg.norm(T[:2,3]-target.position[:2])),
                optical_axis_down_error_deg=float(np.rad2deg(np.arccos(np.clip(-T[2,2],-1.,1.)))),
                refinement_area_ratio=area_ratio)
    result['success']=result['coarse_target_pixel_error']<10. and area_ratio>=.55
    image=packet['rgb'].copy()
    cv2.drawMarker(image,tuple(np.round(principal).astype(int)),(0,255,255),cv2.MARKER_CROSS,26,2)
    cv2.circle(image,tuple(np.round(refined_pixel).astype(int)),9,(255,255,0),2)
    cv2.putText(image,'Yellow: optical center | Cyan: observed target',(20,35),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,0),2)
    cv2.imwrite(str(out/'camera_center_check.png'),image)
    (out/'result.json').write_text(json.dumps(result,indent=2))
    print('CAMERA_VIEW_RESULT '+json.dumps(result),flush=True)
finally:
    backend.hold();backend.evaluator.file.close()
    if backend.controller:
        backend.controller.shutdown()
    backend.destroy_node();rclpy.shutdown()
