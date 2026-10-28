import argparse,json,time,sys,os
from dataclasses import asdict
from pathlib import Path
from planner import RulePlanner,LLMPlanner,DEFAULT_INSTRUCTION,ZONES,HEIGHTS
from orchestrator import execute_plan
from skills import GraspSkills

def main():
    p=argparse.ArgumentParser();p.add_argument('--instruction',default=DEFAULT_INSTRUCTION)
    p.add_argument('--root',default='/home/aluupy/arm_feedback_experiments/20261027')
    p.add_argument('--run-id',default='agent_sort_20261028');p.add_argument('--dry-run',action='store_true')
    p.add_argument('--reset-scene',action='store_true',help='Simulation test setup, once before the whole task')
    p.add_argument('--planner',choices=['rule','llm'],default='rule')
    p.add_argument('--model',default='qwen3.8-chat');p.add_argument('--api-base',default='https://api.llm.ustc.edu.cn/v1')
    p.add_argument('--api-key-stdin',action='store_true');p.add_argument('--record',action='store_true')
    args=p.parse_args();provider=None
    if args.planner=='llm':
        from llm_api import APIPlannerProvider
        key=sys.stdin.readline().strip() if args.api_key_stdin else os.environ.get('ROBOT_LLM_API_KEY','')
        if not key:raise ValueError('Missing runtime API key')
        provider=APIPlannerProvider(args.api_base,args.model,key)
        steps=LLMPlanner(provider).plan(args.instruction)
    else:steps=RulePlanner().plan(args.instruction)
    if not args.run_id.replace('_','').isalnum():p.error('Invalid run ID')
    plan=dict(instruction=args.instruction,planner=args.planner,steps=[asdict(s) for s in steps],zones=ZONES,surface_heights=HEIGHTS,
              llm_response=provider.last_response if provider else None)
    if args.dry_run:print(json.dumps(plan,ensure_ascii=False,indent=2));return 0
    root=Path(args.root);output=root/'agent_runs'/args.run_id;output.mkdir(parents=True,exist_ok=False)
    runtime=json.loads((root/'sim/runtime_config.json').read_text())
    if runtime.get('demo_scene')!='everyday' or runtime.get('gripper_profile')!='short_parallel':
        raise ValueError('This skill configuration requires everyday scene and short_parallel gripper')
    lock=root/'sim/agent_execution.lock';lock.mkdir()
    capturing=False
    try:
        (output/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
        if args.reset_scene:
            (root/'sim/reset_scene_requested').write_text(args.run_id)
            deadline=time.monotonic()+300;ack=root/'sim/reset_scene_ack'
            while not (ack.exists() and ack.read_text()==args.run_id):
                if time.monotonic()>deadline:raise TimeoutError('Scene startup/reset timeout')
                time.sleep(.2)
            time.sleep(1)
        if args.record:
            from recording import recording
            recording(root,args.run_id,'start');capturing=True
        def emit(event):
            event['time']=time.time()
            with (output/'events.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(event)+'\n')
            print(json.dumps(event),flush=True)
        result=execute_plan(steps,GraspSkills(root,args.run_id),emit)
        (output/'result.json').write_text(json.dumps(result,indent=2))
        return 0 if result['success'] else 2
    except Exception as exc:
        (output/'result.json').write_text(json.dumps(dict(success=False,reason=str(exc))))
        raise
    finally:
        try:
            if capturing:
                time.sleep(1)
                status=recording(root,args.run_id,'stop')
                (output/'capture.json').write_text(json.dumps(status,indent=2))
        finally:lock.rmdir()

if __name__=='__main__':raise SystemExit(main())
