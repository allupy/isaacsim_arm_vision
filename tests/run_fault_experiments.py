"""Software-in-loop fault injection. This is NOT a physical grasp benchmark.

Execute the ORIGINAL execute_pick_place function from arm_manager.py, replacing
only motion/actuation interfaces with deterministic responses, and compare its
completion signal with the feedback policy under matching task-result faults.
"""
import ast
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_feedback import SimulatedBackend
from modules.feedback_policy import FeedbackPolicy

ROOT = Path(__file__).resolve().parents[1]


def load_original_method():
    parsed = ast.parse((ROOT/'modules/arm_manager.py').read_text(encoding='utf-8'))
    cls = next(node for node in parsed.body if isinstance(node, ast.ClassDef))
    functions = [node for node in cls.body if isinstance(node, ast.FunctionDef)
                 and node.name in ['execute_pick_place', '_get_grasp_width']]
    namespace = {'List':List, 'Optional':Optional, 'time':SimpleNamespace(sleep=lambda _:None)}
    exec(compile(ast.Module(body=functions, type_ignores=[]), 'original_arm_manager.py', 'exec'), namespace)
    return namespace


class LegacyFixture:
    home_position = [.3,.25,.5]
    default_place_position = [.35,-.35,.05]
    approach_height = .4
    pick_height = .25
    safe_height = .5
    arc_radius = .2
    gripper_open = .8

    def __init__(self):
        logger = SimpleNamespace(info=lambda *_:None, warn=lambda *_:None, error=lambda *_:None)
        self.node = SimpleNamespace(get_logger=lambda:logger)

    def go_home(self, *args, **kwargs):
        return True

    def control_gripper(self, *args, **kwargs):
        return True

    def move_to_position_direct(self, *args, **kwargs):
        return True

    def move_linearly(self, *args, **kwargs):
        return True

    def _arc_move_with_fallback(self, *args, **kwargs):
        return True


def main():
    methods = load_original_method()
    for name in ['execute_pick_place', '_get_grasp_width']:
        setattr(LegacyFixture, name, methods[name])
    output = ROOT/'results'/'fault_injection'
    output.mkdir(parents=True, exist_ok=True)
    scenarios = [('nominal', None, True), ('empty_grasp','empty_grasp',False),
                 ('object_occluded','occluded',None), ('wrong_placement','wrong_place',False)]
    rows = []
    for name, fault, actual_success in scenarios:
        baseline = LegacyFixture().execute_pick_place([.4,.1,.0])
        backend = SimulatedBackend(fault)
        result = FeedbackPolicy(backend, event_sink=backend.event).run()
        (output/f'{name}.json').write_text(json.dumps(result, indent=2),encoding='utf-8')
        rows.append(dict(scenario=name, task_outcome=actual_success,
                         legacy_reports_success=baseline,
                         feedback_reports_success=result['success'],
                         feedback_attempts=result['attempts'], reason=result['reason']))
    summary = {'experiment_type':'deterministic_software_in_loop_not_physics',
               'actuation_contract':'all command interfaces report True in both policies',
               'baseline_source':'modules/arm_manager.py:execute_pick_place (actual method via AST)',
               'scenarios':rows,
               'limitation':'Known synthetic outcomes; no physical success rate or generalization claim.'}
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    with (output/'summary.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
