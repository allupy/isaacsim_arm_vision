"""Run inside the ROS control container; no robot or GPU inference is used."""
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock,patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import feedback_runner

class InterruptTests(unittest.TestCase):
    def test_interrupt_records_whether_hold_was_possible(self):
        for held in (True,False):
            with self.subTest(held=held), tempfile.TemporaryDirectory() as tmp:
                output=Path(tmp)/'result'
                backend=SimpleNamespace(init_detector=Mock(side_effect=KeyboardInterrupt),hold=Mock(return_value=held),
                    writer=SimpleNamespace(shutdown=Mock()),pending_writes=[],evaluator=SimpleNamespace(file=io.StringIO()),
                    controller=None,destroy_node=Mock())
                with patch.object(feedback_runner,'SensorBackend',return_value=backend),patch.object(sys,'argv',['feedback_runner.py','--output',str(output)]):
                    code=feedback_runner.main()
                result=json.loads((output/'result.json').read_text())
                self.assertEqual(code,130)
                self.assertEqual(result['reason'],'interrupted')
                self.assertEqual(result['state_known'],held)
                backend.hold.assert_called_once()
                backend.destroy_node.assert_called_once()

if __name__=='__main__':unittest.main()
