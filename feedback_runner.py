#!/usr/bin/env python3
"""Simulation experiments with sensor evidence separated from ground-truth scoring."""
import argparse
import os
from dataclasses import asdict
import json
from pathlib import Path
import time
import sys
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from config.config import DetectorConfig
from modules.feedback_policy import FeedbackPolicy, PolicyConfig, Observation
from modules.feedback_observation import FrameSynchronizer, mask_geometry, color_masks
from modules.camera_viewpoint import camera_centered_pose


class EvaluationRecorder:
    """Object ground truth is only recorded; it is NEVER an input to the policy."""
    def __init__(self, node, output):
        self.file = (output/'ground_truth.jsonl').open('w')
        self.first = None
        self.last = None
        self.max_height = {}
        self.sub = node.create_subscription(String, '/evaluation/ground_truth', self.record, 10)

    def record(self, msg):
        record = json.loads(msg.data)
        if self.first is None:
            self.first = record
        self.last = record
        for name, xyz in record['objects'].items():
            self.max_height[name] = max(xyz[2], self.max_height.get(name, -float('inf')))
        self.file.write(json.dumps(record)+'\n')
        self.file.flush()

    def summary(self, target_position, destination, requested_label=None, destination_tag=None):
        if self.first is None or self.last is None or target_position is None:
            return {'available': False}
        names = list(self.first['objects'])
        name = min(names, key=lambda n: np.linalg.norm(np.asarray(self.first['objects'][n])[:2]-target_position[:2]))
        selected=name
        requested=(requested_label or '').replace(' ','_')
        if requested in names:
            name=requested
        first = self.first['objects'][name]
        last = self.last['objects'][name]
        lifted = self.max_height[name]-first[2] >= .06
        xy_error = float(np.linalg.norm(np.asarray(last[:2])-destination[:2]))
        # All demo destinations use the same table support height as acquisition.
        # This independent metric also works for non-cube object dimensions.
        height_error = abs(last[2]-first[2])
        xy_limit=.06
        fixture=self.last.get('destinations',{}).get(str(destination_tag)) if destination_tag else None
        if destination_tag and fixture is None:return dict(available=False,reason='destination_truth_missing')
        if fixture:
            xy_error=float(np.linalg.norm(np.asarray(last[:2])-fixture['center'][:2]))
            height_error=abs(last[2]-(first[2]+fixture['center'][2]-.01))
            xy_limit=fixture['radius']-({'apple':.024,'cup':.034}.get(name,.025))
        return dict(available=True, target=name, initial=first, final=last,
                    lifted=lifted, max_lift=self.max_height[name]-first[2],
                    placement_xy_error=xy_error, placement_height_error=height_error,
                    selected_visual_target=selected,requested_target=requested,
                    target_selection_correct=selected==name,
                    destination_fixture=fixture,placement_xy_limit=xy_limit,
                    task_success=bool(selected==name and lifted and xy_error < xy_limit and height_error < (.008 if fixture else .035)))


class SensorBackend(Node):
    def __init__(self, args, output):
        super().__init__('feedback_grasp_experiment')
        self.args, self.output = args, output
        self.bridge = CvBridge()
        self.buffers = {camera: FrameSynchronizer() for camera in ['wrist', 'observer']}
        self.last_stamp = dict(wrist=0, observer=0)
        self.latest_calibration = None
        self.subscriptions_keep = []
        for camera, rgb_topic, depth_topic in [('wrist','/rgb','/depth'), ('observer','/observer_rgb','/observer_depth')]:
            for kind, topic in [('rgb',rgb_topic),('depth',depth_topic)]:
                self.subscriptions_keep.append(self.create_subscription(Image, topic,
                    lambda msg, c=camera, k=kind: self.image_callback(c,k,msg), 3))
        self.calib_sub = self.create_subscription(String, '/feedback/camera_calibration', self.calibration_callback, 3)
        self.camera_request_pub = self.create_publisher(String,'/feedback/camera_request',1)
        self.evaluator = EvaluationRecorder(self, output)
        self.controller = None
        self.detector = None
        self.observation_index = 0
        self.policy = None
        self.tracking_target = None
        self.perturb_done=False
        self.writer = ThreadPoolExecutor(max_workers=1,thread_name_prefix='evidence-writer')
        self.pending_writes = []
        cfg=DetectorConfig.from_yaml(args.config)
        rotation,translation=cfg.get_cam2gripper_pose()
        self.ee_from_camera=np.eye(4)
        self.ee_from_camera[:3,:3]=rotation
        self.ee_from_camera[:3,3]=translation

    def image_callback(self, camera, kind, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, 'bgr8' if kind == 'rgb' else 'passthrough')
            stamp = msg.header.stamp.sec*1000000000+msg.header.stamp.nanosec
            # Drop network backlog and stale replay.
            now_ns = self.get_clock().now().nanoseconds
            if abs(now_ns-stamp) > 8e9:
                return
            self.buffers[camera].add(kind, stamp, image.copy())
        except Exception as exc:
            self.get_logger().error(str(exc))

    def calibration_callback(self, msg):
        calib = json.loads(msg.data)
        self.latest_calibration = calib
        for camera in self.buffers:
            self.buffers[camera].add('calibration', calib['stamp_ns'], calib[camera])

    def pump(self, duration=.05):
        deadline = time.monotonic()+duration
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=.02)

    def init_detector(self):
        if self.args.perception == 'yolo_sam':
            from modules.detection_manager import DetectionManager
            cfg = DetectorConfig.from_yaml(self.args.config)
            cfg.model_path = os.environ.get('ROBOT_YOLO_WEIGHTS', '/ros2_ws/isaacsim_arm_vision/yolov8s-worldv2.pt')
            cfg.sam_checkpoint = os.environ.get('ROBOT_SAM_WEIGHTS', '/ros2_ws/vision/sam_vit_b_01ec64.pth')
            cfg.sam_model_type = 'vit_b'
            cfg.confidence_threshold = self.args.det_confidence
            cfg.default_classes = self.args.classes or ['cube', 'block']
            cfg.max_detections = 8
            if self.args.target_label=='cup':
                cfg.sam_prompt_mode = 'box'
            self.detector = DetectionManager(cfg)

    def wait_ready(self, timeout=180.):
        barrier=self.get_clock().now().nanoseconds
        deadline=time.monotonic()+timeout
        next_request = 0.
        while time.monotonic()<deadline:
            if time.monotonic() >= next_request:
                self.camera_request_pub.publish(String(data='startup'))
                next_request = time.monotonic()+2.
            self.pump()
            if self.buffers['observer'].latest(after_stamp=barrier):
                return True
        return False

    def observe(self, camera):
        started = time.monotonic()
        # Require capture after this call, not just an image arriving after motion.
        barrier = max(self.last_stamp[camera], self.get_clock().now().nanoseconds)
        deadline = time.monotonic()+self.args.sensor_timeout
        packet = None
        next_request = 0.
        while time.monotonic() < deadline:
            if time.monotonic() >= next_request:
                self.camera_request_pub.publish(String(data=camera))
                next_request = time.monotonic()+2.
            self.pump()
            packet = self.buffers[camera].latest(after_stamp=barrier)
            if packet:
                break
        if packet is None:
            self.log({'kind': 'sensor_timeout', 'camera': camera})
            return []
        stamp, data = packet
        self.last_stamp[camera] = stamp
        rgb, depth = data['rgb'], data['depth']
        self.latest_rgbd=(rgb,depth,data['calibration'],stamp)
        wait_wall = time.monotonic()-started
        self.observation_index += 1
        prefix = self.output/f'{self.observation_index:03d}_{camera}'
        def save_evidence():
            cv2.imwrite(str(prefix)+'.png', rgb)
            np.savez_compressed(str(prefix)+'.npz', depth=depth, calibration=json.dumps(data['calibration']))
        if getattr(self.args,'legacy_performance',False):
            save_evidence()
        else:
            self.pending_writes = [f for f in self.pending_writes if not f.done() or f.result()]
            # Bounded backlog: slow disks apply backpressure rather than dropping evidence.
            if len(self.pending_writes) >= 2:
                self.pending_writes.pop(0).result()
            self.pending_writes.append(self.writer.submit(save_evidence))
        candidates = []
        if self.args.perception == 'color':
            masks = [(label, mask, min(1., area/1000.)) for label,mask,area in color_masks(rgb)]
        else:
            annotated, detections = self.detector.detect(rgb, segment_all=True)
            if annotated is not None:
                cv2.imwrite(str(prefix)+'_detections.png', annotated)
            masks = [(detection.class_name, detection.mask, detection.confidence)
                     for detection in detections if detection.mask is not None]
        sources = {id(mask):'hsv' if self.args.perception=='color' else 'yolo_world_sam' for _,mask,_ in masks}
        matched_detection=False
        if self.tracking_target:
            expected,original=self.tracking_target
            for label,mask,_ in masks:
                if label!=original.label: continue
                try:
                    candidate,appearance,_=mask_geometry(rgb,depth,mask,data['calibration'])
                    if (np.linalg.norm(np.asarray(candidate)-expected)<.075 and
                            np.linalg.norm(np.asarray(appearance)-original.appearance)<.35):
                        matched_detection=True
                except ValueError: pass
        if (self.args.locked_sam_tracking and self.detector is not None and self.tracking_target
                and not matched_detection):
            expected, original = self.tracking_target
            tracked = self.segment_locked_region(rgb,depth,data['calibration'],expected,original)
            if tracked is not None:
                masks.append((original.label,tracked,0.))
                sources[id(tracked)]='sam_appearance_depth_tracking_from_visual_lock'
        for label, mask, confidence in masks:
            try:
                source=sources[id(mask)]
                if label=='cup' and self.args.instruction:
                    from modules.feedback_observation import reject_support_pixels
                    mask,plane=reject_support_pixels(depth,mask,data['calibration'])
                position, appearance, quat, extent = mask_geometry(rgb, depth, mask, data['calibration'],return_details=True)
                if label=='cup':
                    # Grasp the upper outside wall; the interior bottom is not a grasp surface.
                    ys,xs=np.where((mask>0)&np.isfinite(depth)&(depth>.03)&(depth<4.))
                    z=depth[ys,xs];K=np.asarray(data['calibration']['K']);T=np.asarray(data['calibration']['optical_to_world'])
                    points=np.column_stack(((xs-K[0,2])*z/K[0,0],(ys-K[1,2])*z/K[1,1],z))@T[:3,:3].T+T[:3,3]
                    position[2]=float(np.percentile(points[:,2],90))
                    rim=points[points[:,2]>=np.percentile(points[:,2],80)]
                    position[:2]=np.median(rim[:,:2],axis=0).tolist()
                # Cubes are symmetric: a stable downward approach avoids noisy PCA yaw.
                if label == 'banana':
                    import math
                    yaw=(2*math.atan2(quat[1],quat[0])+math.pi/2+math.pi/2)%math.pi-math.pi/2
                    quat=[math.cos(yaw/2),math.sin(yaw/2),0.,0.]
                else:
                    quat = [1., 0., 0., 0.]
                candidates.append(Observation(position, appearance, label, confidence, stamp, data['received_at'], quat, extent,source))
            except ValueError as exc:
                self.log({'kind':'invalid_geometry', 'reason':str(exc)})
        self.log({'kind':'observation', 'camera':camera, 'stamp_ns':stamp,
                  'perception':self.args.perception, 'wait_wall_s':wait_wall,
                  'processing_wall_s':time.monotonic()-started-wait_wall,
                  'candidates':[asdict(c) for c in candidates]})
        return candidates

    def set_tracking_target(self, expected, original):
        self.tracking_target=(expected,original)

    def segment_locked_region(self,rgb,depth,calibration,expected,original):
        """Re-segment a visual lock near its commanded pose; never read simulator truth."""
        sam=self.detector.sam_predictor
        if sam is None: return None
        from itertools import product
        center=np.asarray(expected,dtype=float).copy();center[2]-=.02
        extent=np.asarray(original.extent_xy or [.05,.05])
        half=np.r_[np.clip(extent*.65,.02,.10),.032]
        points=np.array([center+np.asarray(sign)*half for sign in product([-1,1],repeat=3)])
        inv=np.linalg.inv(np.asarray(calibration['optical_to_world']))
        local=points@inv[:3,:3].T+inv[:3,3]
        if np.any(local[:,2]<=.03): return None
        K=np.asarray(calibration['K']);pixels=local@K.T;pixels=pixels[:,:2]/pixels[:,2,None]
        low,high=np.min(pixels,axis=0),np.max(pixels,axis=0)
        h,w=rgb.shape[:2]
        box=np.r_[np.clip(low,[0,0],[w-1,h-1]),np.clip(high,[0,0],[w-1,h-1])]
        if box[2]-box[0]<4 or box[3]-box[1]<4: return None
        # Positive cues must resemble the ORIGINAL visual lock and have compatible
        # measured depth; a gripper or background patch cannot replace the target.
        x1,y1,x2,y2=map(int,box)
        yy,xx=np.mgrid[y1:y2+1,x1:x2+1]
        zz=depth[yy,xx]
        colors=rgb[yy,xx,::-1].astype(float)
        chroma=colors/np.maximum(colors.sum(axis=2,keepdims=True),1.)
        delta=np.linalg.norm(chroma-np.asarray(original.appearance),axis=2)
        safe_z=np.nan_to_num(zz,nan=0.,posinf=0.,neginf=0.)
        xyz=np.stack(((xx-K[0,2])*safe_z/K[0,0],(yy-K[1,2])*safe_z/K[1,1],safe_z),axis=-1)
        T=np.asarray(calibration['optical_to_world']);world=xyz@T[:3,:3].T+T[:3,3]
        compatible=(np.isfinite(zz)&(zz>.03)&(delta<.13)&
                    (np.linalg.norm(world-np.asarray(expected),axis=2)<.09))
        if compatible.sum()<6: return None
        distance=cv2.distanceTransform(compatible.astype(np.uint8),cv2.DIST_L2,3)
        py,px=np.unravel_index(np.argmax(distance),distance.shape)
        point_coords=np.array([[x1+px,y1+py]],dtype=float);point_labels=np.array([1])
        # DetectionManager already encoded this same image if any detection exists.
        sam.set_image(cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB))
        masks,_,_=sam.predict(box=box,point_coords=point_coords,point_labels=point_labels,multimask_output=False)
        self.log(dict(kind='locked_sam_tracking',label=original.label,expected=expected,box=box.tolist(),
                      source='prior visual lock and commanded motion; RGB segmentation and depth verification'))
        # SAM sometimes groups the contacted object with the fingers. Retain only
        # original-appearance pixels with compatible measured 3D positions.
        tracked=np.zeros_like(masks[0],dtype=np.uint8)
        tracked[yy,xx]=masks[0][yy,xx]&compatible
        return tracked

    def log(self, record):
        if record.get('locked_target') and self.args.test_perturb and not self.perturb_done:
            self.perturb_done=True
            request=dict(id=self.output.name,object='red_cube',offset=[-.035,0.,0.])
            path=Path('/experiment/sim/perturb_request.json')
            temp=path.with_suffix('.tmp');temp.write_text(json.dumps(request));temp.replace(path)
            deadline=time.monotonic()+10
            ack=path.parent/'perturb_ack.txt'
            while not (ack.exists() and ack.read_text()==self.output.name):
                if time.monotonic()>deadline:raise TimeoutError('Perturbation fixture acknowledgement missing')
                self.pump(.05)
            self.pump(.4)
            record['test_fixture']='Target displaced 35 mm after visual lock; new pose NOT supplied to policy'
        record['time'] = time.time()
        if self.controller is not None and self.controller.sim_time is not None:
            record['sim_time'] = self.controller.sim_time
        with (self.output/'events.jsonl').open('a') as stream:
            stream.write(json.dumps(record)+'\n')
        print(json.dumps(record), flush=True)

    def move(self, position, quat, linear=False):
        started = time.monotonic()
        if self.latest_calibration is None:
            return False
        # The calibration frame is simulator world. MoveIt world is the URDF base.
        transform = np.asarray(self.latest_calibration.get('planning_to_world', np.eye(4)))
        from scipy.spatial.transform import Rotation
        inv = np.linalg.inv(transform)
        local_pos = (inv @ np.r_[position,1.])[:3].tolist()
        local_q = (Rotation.from_matrix(inv[:3,:3]) * Rotation.from_quat(quat)).as_quat().tolist()
        if (not getattr(self.args,'legacy_performance',False) and self.controller.motion_phase == 'HOME' and
                abs(time.time()-self.latest_calibration['stamp_ns']/1e9) < .5):
            measured = np.asarray(self.latest_calibration['wrist']['optical_to_world']) @ np.linalg.inv(self.ee_from_camera)
            position_error = float(np.linalg.norm(measured[:3,3]-position))
            angle_error = float((Rotation.from_matrix(measured[:3,:3]).inv()*Rotation.from_quat(quat)).magnitude())
            if position_error < .003 and angle_error < .02:
                self.log(dict(kind='motion',goal_world=position,skipped=True,reason='fresh_pose_already_at_home',
                              position_error=position_error,angle_error=angle_error,wall_s=time.monotonic()-started))
                return True
        ok = (self.controller.cartesian_to(local_pos, local_q) if linear else
              self.controller.move_to_position_direct(local_pos, orientation=local_q))
        self.log({'kind':'motion', 'goal_world':position, 'goal_planning':local_pos,
                  'linear':linear, 'wall_s':time.monotonic()-started,'result':self.controller.last_motion})
        return ok

    def set_motion_phase(self, phase):
        self.controller.motion_phase = phase

    def observation_pose(self,target_world,camera_clearance):
        from scipy.spatial.transform import Rotation
        world_ee,world_camera=camera_centered_pose(target_world,camera_clearance,self.ee_from_camera)
        self.log(dict(kind='camera_view_goal',target_world=target_world,camera_clearance=camera_clearance,
                      world_from_camera=world_camera.tolist(),world_from_ee=world_ee.tolist(),
                      ee_from_camera=self.ee_from_camera.tolist()))
        return world_ee[:3,3].tolist(),Rotation.from_matrix(world_ee[:3,:3]).as_quat().tolist()

    def grip(self, closed):
        ok = self.controller.set_gripper(closed)
        self.log({'kind':'gripper', 'closed':closed, 'success':ok})
        return ok

    def hold(self):
        if self.controller is not None:
            held = self.controller.hold()
            self.pump(.2)
            return held
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--perception', choices=['yolo_sam', 'color'], default='yolo_sam')
    parser.add_argument('--config', default='config/default_config.yaml')
    parser.add_argument('--observe-only', action='store_true')
    parser.add_argument('--sensor-timeout', type=float, default=25.)
    parser.add_argument('--startup-timeout',type=float,default=180.)
    parser.add_argument('--tcp-offset', type=float, default=.20)
    parser.add_argument('--observation-mode',choices=['camera_centered','legacy_offset'],default='camera_centered')
    parser.add_argument('--camera-clearance',type=float,default=.38)
    parser.add_argument('--trajectory-time-scale', type=float, default=1.)
    parser.add_argument('--legacy-performance',action='store_true')
    parser.add_argument('--safe-z',type=float,default=.5)
    parser.add_argument('--target-label')
    parser.add_argument('--destination-tag',type=int,choices=[10,20])
    parser.add_argument('--acquire-camera',choices=['wrist','observer'])
    parser.add_argument('--test-perturb',action='store_true',help='Simulator experiment ONLY: move red cube after visual lock')
    parser.add_argument('--instruction', help='One target, controlled Chinese/English demo command')
    parser.add_argument('--classes', nargs='+')
    parser.add_argument('--det-confidence', type=float, default=.02)
    parser.add_argument('--locked-sam-tracking',action='store_true')
    parser.add_argument('--expected-gripper',choices=['robotiq_2f140','short_parallel'])
    parser.add_argument('--home', type=float, nargs=3, default=[.30,.25,.50])
    parser.add_argument('--place-surface', type=float, nargs=3, default=[.35,-.35,.05])
    args = parser.parse_args()
    command = None
    if args.instruction:
        from modules.demo_command import parse_command
        command = parse_command(args.instruction)
        if args.target_label and args.target_label != command['target']:
            parser.error('Instruction and target label disagree')
        args.target_label = command['target']
        if not args.classes:
            args.classes = (['red cube','green cube','blue cube'] if 'cube' in args.target_label
                            else ['cup','banana','apple'])
    output = Path(args.output)
    if args.test_perturb:
        if args.target_label!='red cube':parser.error('Perturbation fixture only supports the red cube')
        command['display']+=' | TEST: target shifted 35 mm after lock'
    output.mkdir(parents=True, exist_ok=False)
    from rclpy.signals import SignalHandlerOptions
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    backend = SensorBackend(args, output)
    code = 1
    try:
        backend.init_detector()
        if not backend.wait_ready(args.startup_timeout):
            raise TimeoutError('startup_sensor_not_ready')
        if args.expected_gripper and backend.latest_calibration.get('gripper',{}).get('id')!=args.expected_gripper:
            raise ValueError('Unexpected simulator gripper profile')
        if args.observe_only:
            observations = {c: [asdict(o) for o in backend.observe(c)] for c in ['wrist','observer']}
            result = {'mode':'observe_only','perception':args.perception,'observations':observations}
            code = 0 if any(observations.values()) else 2
        else:
            from modules.feedback_motion import FeedbackArmController
            backend.controller = FeedbackArmController(backend, params={'trajectory_time_scale':args.trajectory_time_scale,
                                                                       'optimized':not args.legacy_performance})
            # Acquire calibration before moving; this does not select a target.
            if args.destination_tag:
                from modules.destination_vision import locate_support
                supports=[]
                last_support_stamp=0
                for _ in range(5):
                    backend.observe('observer')
                    try:
                        rgb,depth,calib,stamp=backend.latest_rgbd
                        if stamp<=last_support_stamp or abs(backend.get_clock().now().nanoseconds-stamp)>8e9:
                            raise ValueError('destination_frame_stale')
                        support,corners=locate_support(rgb,depth,calib,args.destination_tag)
                        last_support_stamp=stamp
                        if supports and np.linalg.norm(np.asarray(support)-supports[-1])>.005:supports=[]
                        supports.append(support)
                        backend.log(dict(kind='destination_observed',tag=args.destination_tag,support=support,corners=corners,stamp_ns=stamp))
                        if len(supports)>=2:break
                    except ValueError as exc:backend.log(dict(kind='destination_rejected',reason=str(exc)))
                if len(supports)<2:raise ValueError('Destination was not stably localized')
                support=np.mean(supports,axis=0)
                args.place_surface=[float(support[0]),float(support[1]),float(support[2]+{'apple':.042,'cup':.048}[args.target_label])]
            if args.legacy_performance:
                backend.observe('observer')
            cfg = PolicyConfig(tcp_offset=args.tcp_offset,home=args.home,place_surface=args.place_surface,
                               observation_mode=args.observation_mode,camera_clearance=args.camera_clearance,
                               optimized_paths=not args.legacy_performance,safe_z=args.safe_z,target_label=args.target_label)
            if command:
                # The demo operates on the visible tabletop, away from the arm base.
                cfg.workspace_min=[.20,-.50,.016]
                cfg.workspace_max=[.80,.40,.15]
                if args.target_label=='cup':
                    cfg.acquire_camera='observer'
            if args.acquire_camera:
                cfg.acquire_camera=args.acquire_camera
            policy = FeedbackPolicy(backend, cfg, backend.log)
            backend.policy = policy
            result = policy.run()
            result['config'] = asdict(cfg)
            result['perception'] = args.perception
            result['gripper'] = backend.latest_calibration.get('gripper',{})
            backend.pump(2.)
            result['ground_truth_evaluation'] = backend.evaluator.summary(
                policy.lock.original.position if policy.lock else None, cfg.place_surface,args.target_label,args.destination_tag)
            code = 0 if result['success'] else 2
        result['command'] = command
        result['detector_classes'] = args.classes
        result['detector_confidence'] = args.det_confidence
        result['locked_sam_tracking'] = args.locked_sam_tracking
        result['test_perturb'] = args.test_perturb
        (output/'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print('EXPERIMENT_RESULT '+json.dumps(result), flush=True)
    except (Exception, KeyboardInterrupt) as exc:
        import traceback
        traceback.print_exc()
        held = False
        try:
            held = bool(backend.hold())
        finally:
            reason = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'runtime_error:'+str(exc)
            code = 130 if isinstance(exc, KeyboardInterrupt) else 1
            (output/'result.json').write_text(json.dumps({'success':False,'reason':reason,'state_known':held},indent=2))
    finally:
        backend.writer.shutdown(wait=True)
        for future in backend.pending_writes:
            future.result()
        backend.evaluator.file.close()
        if backend.controller is not None:
            backend.controller.shutdown()
        backend.destroy_node()
        rclpy.shutdown()
    return code


if __name__ == '__main__':
    sys.exit(main())
