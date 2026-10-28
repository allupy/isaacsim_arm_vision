import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import time
import unittest
import numpy as np
from modules.feedback_policy import FeedbackPolicy, PolicyConfig, Observation, TargetLock, PolicyFailure, State
from modules.feedback_observation import FrameSynchronizer, mask_geometry


def observation(position=None, appearance=None, confidence=.6, stamp=1):
    return Observation(position or [.4,.1,.05], appearance or [.8,.1,.1],
                       'cube', confidence, stamp, time.monotonic())


class SimulatedBackend:
    """Deterministic contract fixture, not an Isaac Sim or physical success test."""
    def __init__(self, fault=None):
        self.fault = fault
        self.state = None
        self.attempt = 0
        self.position = [.4,.1,.05]
        self.ee = [.3,.25,.5]
        self.closed = False
        self.held = False
        self.stamp = 0
        self.hold_count = 0
        self.release_after_lift = 0
        self.moves = []
        self.goals = []
        self.observe_count = 0

    def event(self, event):
        self.state = event['state']
        self.attempt = event['attempt']

    def move(self, position, quat, linear=False):
        self.moves.append(self.state)
        self.goals.append((self.state,list(position),list(quat)))
        if self.fault == 'motion_failure' and self.state == 'DESCEND':
            return False
        if self.fault == 'transport_failure' and self.state == 'TRANSFER':
            return False
        if self.held:
            self.position = [p+(new-old) for p,new,old in zip(self.position,position,self.ee)]
        self.ee = list(position)
        return True

    def observation_pose(self,target,clearance):
        from modules.camera_viewpoint import camera_centered_pose
        from scipy.spatial.transform import Rotation
        mount=np.eye(4);mount[:3,3]=[.012674,-.116914,-.000172]
        ee,_=camera_centered_pose(target,clearance,mount)
        return ee[:3,3].tolist(),Rotation.from_matrix(ee[:3,:3]).as_quat().tolist()

    def grip(self, closed):
        self.closed = closed
        if closed:
            self.held = self.fault != 'empty_grasp'
        else:
            if self.held:
                self.release_after_lift += 1
                self.position[2] = .05
                if self.fault == 'wrong_place':
                    self.position[0] += .18
            self.held = False
        return True

    def observe(self, camera):
        self.observe_count += 1
        self.stamp += 1
        if self.fault == 'missing' or (self.fault == 'refine_retry' and self.state == 'REFINE' and self.attempt == 1):
            return []
        if self.fault == 'occluded' and self.state == 'VERIFY_LIFT':
            return []
        stamp = 1 if self.fault == 'stale' else self.stamp
        obj = observation(self.position.copy(), stamp=stamp)
        obj.extent_xy = [.05,.05]
        if self.fault == 'partial_mask' and self.state == 'REFINE':
            obj.extent_xy = [.005,.05]
        if self.fault == 'distractor' and self.state != 'ACQUIRE':
            return [observation([.41,.11,.05], [.1,.8,.1], .99, stamp), obj]
        return [obj]

    def hold(self):
        self.hold_count += 1


class PolicyTests(unittest.TestCase):
    def run_policy(self, fault=None):
        backend = SimulatedBackend(fault)
        policy = FeedbackPolicy(backend, event_sink=backend.event)
        result = policy.run()
        return backend, result

    def test_complete_requires_lift_and_place_evidence(self):
        backend, result = self.run_policy()
        self.assertTrue(result['success'])
        self.assertTrue(result['lift_verified'])
        evidence = [e for e in result['events'] if 'evidence' in e]
        self.assertEqual(len(evidence), 4)

    def test_empty_grasp_is_not_success(self):
        backend, result = self.run_policy('empty_grasp')
        self.assertFalse(result['success'])
        self.assertNotIn('TRANSFER', backend.moves)
        self.assertEqual(result['attempts'], 2)

    def test_occlusion_is_not_success(self):
        _, result = self.run_policy('occluded')
        self.assertFalse(result['success'])
        self.assertFalse(result['lift_verified'])

    def test_wrong_place_is_not_success(self):
        _, result = self.run_policy('wrong_place')
        self.assertTrue(result['lift_verified'])
        self.assertFalse(result['success'])

    def test_missing_is_bounded(self):
        backend, result = self.run_policy('missing')
        self.assertFalse(result['success'])
        self.assertEqual(backend.observe_count, 3)

    def test_incomplete_refinement_does_not_drive_grasp(self):
        backend,result=self.run_policy('partial_mask')
        self.assertFalse(result['success'])
        self.assertNotIn('DESCEND',backend.moves)
        self.assertEqual(result['reason'],'refinement_incomplete_mask')

    def test_stale_does_not_refine(self):
        backend, result = self.run_policy('stale')
        self.assertFalse(result['success'])
        self.assertNotIn('DESCEND', backend.moves)

    def test_more_confident_distractor_does_not_switch(self):
        _, result = self.run_policy('distractor')
        self.assertTrue(result['success'])

    def test_refine_failure_recovers_once(self):
        _, result = self.run_policy('refine_retry')
        self.assertTrue(result['success'])
        self.assertEqual(result['attempts'], 2)

    def test_motion_failure_stops(self):
        backend, result = self.run_policy('motion_failure')
        self.assertFalse(result['success'])
        self.assertNotIn('LIFT', backend.moves)
        self.assertEqual(backend.hold_count, 1)

    def test_transport_failure_does_not_release(self):
        backend, result = self.run_policy('transport_failure')
        self.assertFalse(result['success'])
        self.assertTrue(result['lift_verified'])
        self.assertEqual(backend.release_after_lift, 0)
        self.assertTrue(backend.closed)

    def test_ambiguous_identity_is_rejected(self):
        lock = TargetLock(observation(), PolicyConfig())
        with self.assertRaisesRegex(PolicyFailure, 'ambiguous'):
            lock.match([observation([.401,.1,.05], stamp=2), observation([.402,.1,.05], stamp=2)], [.4,.1,.05])

    def test_appearance_does_not_drift(self):
        lock = TargetLock(observation(), PolicyConfig())
        lock.match([observation(appearance=[.6,.3,.1], stamp=2)], [.4,.1,.05])
        with self.assertRaises(PolicyFailure):
            lock.match([observation(appearance=[.4,.5,.1], stamp=3)], [.4,.1,.05])


class GeometryTests(unittest.TestCase):
    def test_camera_goal_maps_back_to_overhead_optical_pose(self):
        from modules.camera_viewpoint import camera_centered_pose
        from scipy.spatial.transform import Rotation
        mount=np.eye(4)
        mount[:3,:3]=Rotation.from_euler('xyz',[.2,-.3,.1]).as_matrix()
        mount[:3,3]=[.03,-.12,.04]
        target=np.array([.55,-.15,.06])
        ee,camera=camera_centered_pose(target,.38,mount,yaw=.7)
        np.testing.assert_allclose(ee@mount,camera,atol=1e-12)
        in_camera=np.linalg.inv(camera)@np.r_[target,1.]
        np.testing.assert_allclose(in_camera[:3],[0,0,.38],atol=1e-12)

    def test_camera_goal_uses_mount_offset_and_rejects_bad_calibration(self):
        from modules.camera_viewpoint import camera_centered_pose
        mount=np.eye(4);mount[:3,3]=[.012674,-.116914,-.000172]
        ee,_=camera_centered_pose([.55,-.15,.06],.38,mount)
        np.testing.assert_allclose(ee[:3,3],[.537326,-.266914,.439828])
        mount[0,0]=2.
        with self.assertRaises(ValueError):
            camera_centered_pose([.55,-.15,.06],.38,mount)

    def test_policy_observation_uses_camera_center_instead_of_fixed_side_step(self):
        backend=SimulatedBackend()
        result=FeedbackPolicy(backend,event_sink=backend.event).run()
        self.assertTrue(result['success'])
        view=next(goal for state,goal,_ in backend.goals if state=='APPROACH')
        np.testing.assert_allclose(view,[.387326,-.016914,.429828])

    def test_ros_ik_adapter_returns_all_joints_in_requested_order(self):
        # Execute the real adapter method with only the ROS service boundary
        # replaced, so this regression can run without a ROS installation.
        import ast
        from types import SimpleNamespace as NS
        source=(Path(__file__).resolve().parents[1]/'modules/feedback_motion.py').read_text()
        cls=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef))
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='compute_ik_single')
        def request():
            return NS(ik_request=NS(pose_stamped=NS(header=NS(),pose=NS(position=NS(),orientation=NS())),
                                   robot_state=NS(joint_state=NS()),timeout=NS()))
        namespace={'GetPositionIK':NS(Request=request)}
        exec(compile(ast.Module(body=[method],type_ignores=[]),'feedback_motion.py','exec'),namespace)
        names=['a','b','c','d','e','f','finger']
        response=NS(error_code=NS(val=1),solution=NS(joint_state=NS(name=list('fedcba'),position=[6.,5.,4.,3.,2.,1.])))
        controller=NS(joint_names=names,ik_client=None,call=lambda *args:response)
        q=namespace['compute_ik_single'](controller,[.4,0.,.5],[1.,0.,0.,0.],[0.]*6)
        self.assertEqual(q,[1.,2.,3.,4.,5.,6.])

    def test_ik_equivalent_branch_stays_near_current_joint(self):
        from modules.joint_geometry import nearest_equivalent
        import math
        q=nearest_equivalent(-1.57,4.71,-2*math.pi,2*math.pi)
        self.assertLess(abs(q-4.71),.01)
        q=nearest_equivalent(3.118,-1.70,-2*math.pi,2*math.pi)
        self.assertLess(abs(q+3.165),.01)
        with self.assertRaises(ValueError):
            nearest_equivalent(2.,0.,-.5,.5)

    def test_yaml_loads_controller_and_two_stage_fields(self):
        from config.config import DetectorConfig
        cfg = DetectorConfig.from_yaml(str(Path(__file__).resolve().parents[1]/'config/default_config.yaml'))
        self.assertEqual(cfg.controller, 'moveit')
        self.assertTrue(cfg.two_stage_grasp)
        self.assertTrue(cfg.record_video)

    def test_timestamp_matching_rejects_mixed_frames(self):
        sync = FrameSynchronizer()
        sync.add('rgb', 1, 'rgb')
        sync.add('depth', 2, 'depth')
        sync.add('calibration', 1, {})
        self.assertIsNone(sync.latest())
        sync.add('depth', 1, 'depth')
        self.assertEqual(sync.latest()[0], 1)
        self.assertIsNone(sync.latest(after_stamp=1))

    def test_stale_and_unbounded_buffer(self):
        sync = FrameSynchronizer(capacity=2)
        for stamp in range(4):
            for kind in ['rgb','depth','calibration']:
                sync.add(kind, stamp, 0, received_at=time.monotonic()-30)
        self.assertEqual(len(sync.frames), 2)
        self.assertIsNone(sync.latest())

    def test_invalid_centroid_depth_uses_other_mask_pixels(self):
        rgb = np.zeros((20,20,3), dtype=np.uint8)
        rgb[:,:,2] = 200
        depth = np.full((20,20), .5, dtype=float)
        depth[10,10] = np.nan
        depth[9,9] = 4.
        mask = np.zeros((20,20),dtype=np.uint8)
        mask[5:15,5:15] = 1
        calibration = {'K':[[100,0,10],[0,100,10],[0,0,1]], 'optical_to_world':np.eye(4).tolist()}
        pos, appearance, quat = mask_geometry(rgb,depth,mask,calibration)
        self.assertAlmostEqual(pos[2], .5)
        self.assertGreater(appearance[0], .9)

    def test_all_invalid_depth_rejected(self):
        with self.assertRaises(ValueError):
            mask_geometry(np.zeros((20,20,3),np.uint8),np.zeros((20,20)),np.ones((20,20)),{})


if __name__ == '__main__':
    unittest.main(verbosity=2)
