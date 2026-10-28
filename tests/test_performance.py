import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
import numpy as np
from modules.trajectory_playback import PhysicsTrajectory, retime_cartesian
from modules.feedback_policy import FeedbackPolicy, PolicyConfig
from test_feedback import SimulatedBackend


class PlaybackTests(unittest.TestCase):
    def test_hermite_endpoint_velocity_and_limit_overshoot(self):
        player=PhysicsTrajectory(['a'],[0.,1.],[[0.],[1.]],0.,{'a':(-2.,2.)},velocities=[[0.],[0.]])
        self.assertAlmostEqual(player.sample(.5)[0][0],.5)
        self.assertLess(player.sample(.001)[0][0]/.001,.004)
        self.assertLess((1-player.sample(.999)[0][0])/.001,.004)
        with self.assertRaises(ValueError):
            PhysicsTrajectory(['a'],[0.,1.],[[0.],[0.]],0.,{'a':(-1.,1.)},velocities=[[10.],[-10.]])
    def test_render_pause_does_not_advance_or_skip_sim_samples(self):
        player=PhysicsTrajectory(['a'],[0.,1.],[[0.],[1.]],10.,{'a':(-2.,2.)})
        previous=None
        for t in np.arange(10.,11.+1e-6,1/120):
            q,done=player.sample(t)
            if previous is not None:
                self.assertLessEqual(abs(q[0]-previous),1/120+1e-8)
            self.assertTrue(np.array_equal(q,player.sample(t)[0]))
            previous=q[0]
        np.testing.assert_allclose(player.sample(12.)[0],[1.])

    def test_malformed_and_out_of_limit_trajectory_rejected(self):
        for times,positions in [([0.,0.],[[0.],[1.]]),([0.,1.],[[0.],[3.]]),
                                ([0.,1.],[[0.],[float('nan')]])]:
            with self.assertRaises(ValueError):
                PhysicsTrajectory(['a'],times,positions,0.,{'a':(-2.,2.)})

    def test_retiming_limits_and_path_envelope(self):
        original=np.array([[0.,0.],[.1,.03],[.2,.08],[.3,.12]])
        t,q=retime_cartesian(original,velocity=.4,acceleration=.7)
        t,q=np.asarray(t),np.asarray(q)
        dt=t[1]-t[0]
        self.assertLessEqual(np.max(np.abs(np.diff(q,axis=0)/dt)),.4)
        self.assertLessEqual(np.max(np.abs(np.diff(q,n=2,axis=0)/dt**2)),.7)
        np.testing.assert_allclose(q[[0,-1]],original[[0,-1]])
        self.assertTrue(np.all(q>=original.min(axis=0)))
        self.assertTrue(np.all(q<=original.max(axis=0)))
        self.assertLess(np.max(np.abs(q[1]-q[0]))/dt,.01)

    def test_paths_keep_verification_and_remove_one_transfer_stop(self):
        counts=[]
        for optimized in [False,True]:
            backend=SimulatedBackend()
            policy=FeedbackPolicy(backend,PolicyConfig(optimized_paths=optimized),backend.event)
            result=policy.run()
            self.assertTrue(result['success'],result['reason'])
            self.assertEqual(len([e for e in result['events'] if 'evidence' in e]),4)
            counts.append(backend.moves.count('TRANSFER'))
        self.assertEqual(counts,[2,1])


if __name__=='__main__':
    unittest.main()
