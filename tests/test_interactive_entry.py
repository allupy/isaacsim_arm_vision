import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('interactive_main',ROOT/'main.py')
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)

class InteractiveTests(unittest.TestCase):
    def make(self,root,*args):
        return entry.Console(entry.parser().parse_args(['--runtime-root',str(root),*args]))

    def test_demo_parameters_use_same_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'sim').mkdir()
            (root/'sim/runtime_config.json').write_text(json.dumps(dict(demo_scene='everyday',gripper_profile='short_parallel')))
            console=self.make(root)
            cmd=console.command(root/'run')
            self.assertEqual(Path(cmd[2]).name,'feedback_runner.py')
            self.assertEqual(cmd[cmd.index('--instruction')+1],'Pick the apple')
            self.assertEqual(cmd[cmd.index('--home')+1:cmd.index('--home')+4],['0.45','-0.12','0.5'])
            self.assertEqual(cmd[cmd.index('--place-surface')+1:cmd.index('--place-surface')+4],['0.35','-0.3','0.052'])
            self.assertIn('--locked-sam-tracking',cmd)

    def test_observe_only_cannot_be_overridden_by_grasp_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            console=self.make(tmp,'--observe-only')
            self.assertIn('--observe-only',console.command(Path(tmp)/'run'))

    def test_target_destination_and_custom_surface(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=self.make(tmp)
            c.dispatch('t 杯子');c.dispatch('p cup_stand')
            cmd=c.command(Path(tmp)/'run')
            self.assertEqual(cmd[cmd.index('--destination-tag')+1],'20')
            c.dispatch('p .4 -.2 .06')
            self.assertNotIn('--destination-tag',c.command(Path(tmp)/'run'))
            with self.assertRaises(ValueError):c.dispatch('p nan 0 0')
            with self.assertRaises(ValueError):c.dispatch('t unknown')

    def test_shared_agent_lock_is_not_removed_when_owned_by_another(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=self.make(tmp)
            with c.lock():
                with self.assertRaises(RuntimeError):
                    with c.lock():pass
                self.assertTrue((Path(tmp)/'sim/agent_execution.lock').exists())
            self.assertFalse((Path(tmp)/'sim/agent_execution.lock').exists())

    def test_unknown_state_blocks_new_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=self.make(tmp);c.blocked=True
            with patch.object(entry.subprocess,'Popen') as popen:
                with self.assertRaises(RuntimeError):c.run()
                popen.assert_not_called()

if __name__=='__main__':unittest.main()
