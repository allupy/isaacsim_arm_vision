import unittest
from planner import RulePlanner,LLMPlanner,DEFAULT_INSTRUCTION
from orchestrator import execute_plan
class Tests(unittest.TestCase):
    def test_plan_order(self):
        self.assertEqual([(s.target,s.destination) for s in RulePlanner().plan(DEFAULT_INSTRUCTION)],
                         [('apple','fruit_zone'),('cup','cup_zone')])
    def test_reject_unknown_and_occupied(self):
        for text in ['把香蕉放到水果区','把苹果放到水果区，再把杯子放到水果区']:
            with self.assertRaises(ValueError):RulePlanner().plan(text)
    def test_llm_validation(self):
        with self.assertRaises(RuntimeError):LLMPlanner().plan('hello')
        with self.assertRaises(ValueError):LLMPlanner(lambda *_:[dict(target='apple',destination='unknown',skill='move')]).plan('hello')
    def test_stop_on_failure(self):
        class Skills:
            calls=0
            def pick_and_place(self,*args):self.calls+=1;return dict(success=False,reason='not_lifted')
        skills=Skills();result=execute_plan(RulePlanner().plan(DEFAULT_INSTRUCTION),skills,lambda e:None)
        self.assertFalse(result['success']);self.assertEqual(skills.calls,1)
if __name__=='__main__':unittest.main()
