"""Ensure a user's agent targets and stops only its configured container."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'agent_workflow'))
from planner import RulePlanner, DEFAULT_INSTRUCTION
from skills import GraspSkills


class CollaborationConfigTests(unittest.TestCase):
    def test_custom_container_executes_skill(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, CONTROL_NAME='arm-drs-control'):
            root = Path(tmp)
            output = root / 'runs/demo_1'
            output.mkdir(parents=True)
            (output / 'result.json').write_text(json.dumps({'success': True}))
            with patch('skills.subprocess.run', return_value=subprocess.CompletedProcess([], 0)) as run:
                result = GraspSkills(root, 'demo').pick_and_place(RulePlanner().plan(DEFAULT_INSTRUCTION)[0], 1)
            self.assertTrue(result['success'])
            self.assertEqual(run.call_args.args[0][:3], ['docker', 'exec', 'arm-drs-control'])

    def test_timeout_stops_only_custom_container(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, CONTROL_NAME='arm-drs-control'):
            root = Path(tmp)
            (root / 'runs').mkdir()
            with patch('skills.subprocess.run', side_effect=[subprocess.TimeoutExpired('docker', 240), subprocess.CompletedProcess([], 0)]) as run:
                result = GraspSkills(root, 'demo').pick_and_place(RulePlanner().plan(DEFAULT_INSTRUCTION)[0], 1)
            self.assertFalse(result['state_known'])
            self.assertEqual(run.call_args.args[0], ['docker', 'stop', '-t', '3', 'arm-drs-control'])


if __name__ == '__main__':
    unittest.main()
