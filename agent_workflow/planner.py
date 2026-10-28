"""Planner boundary: both deterministic and future LLM plans share validation."""
import re
from dataclasses import dataclass,asdict

ZONES={'fruit_zone':(.35,-.30),'cup_zone':(.49,-.30),'plate':None,'cup_stand':None}
HEIGHTS={'apple':.052,'cup':.058}
DEFAULT_INSTRUCTION='把苹果放到水果区，再把杯子放到杯子区'

@dataclass(frozen=True)
class Step:
    target:str
    destination:str
    skill:str='pick_and_place'

def validate_plan(data):
    if not isinstance(data,list) or not 1<=len(data)<=6:raise ValueError('Plan must contain 1-6 steps')
    steps=[]
    for row in data:
        if set(row)!={'target','destination','skill'}:raise ValueError('Unexpected plan fields')
        step=Step(**row)
        if step.skill!='pick_and_place' or step.target not in HEIGHTS or step.destination not in ZONES:
            raise ValueError('Unsupported skill, object or zone')
        if any(s.destination==step.destination for s in steps):raise ValueError('Zone already reserved by this plan')
        steps.append(step)
    return steps

class RulePlanner:
    def plan(self,instruction):
        clauses=re.split(r'[，,；;]|再|然后',instruction)
        rows=[]
        for clause in clauses:
            clause=clause.strip(' 。！!')
            if not clause:continue
            match=re.fullmatch(r'把(苹果|杯子)放到(水果区|杯子区)',clause)
            if not match:raise ValueError('Unsupported instruction clause: '+clause)
            rows.append(dict(target={'苹果':'apple','杯子':'cup'}[match[1]],
                             destination={'水果区':'fruit_zone','杯子区':'cup_zone'}[match[2]],skill='pick_and_place'))
        return validate_plan(rows)

class LLMPlanner:
    """Inject a provider callable: request_plan(instruction, allowed_schema) -> list.
    API credentials and transport belong to the provider; nothing is called implicitly.
    """
    def __init__(self,request_plan=None):self.request_plan=request_plan
    def plan(self,instruction):
        if self.request_plan is None:raise RuntimeError('LLM API provider is not configured')
        schema=dict(targets=list(HEIGHTS),destinations=list(ZONES),skills=['pick_and_place'],max_steps=6)
        return validate_plan(self.request_plan(instruction,schema))
