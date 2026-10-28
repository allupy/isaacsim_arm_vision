from dataclasses import asdict

def execute_plan(steps,skills,emit):
    results=[]
    for index,step in enumerate(steps,1):
        emit(dict(event='step_started',index=index,step=asdict(step)))
        try:result=skills.pick_and_place(step,index)
        except Exception as exc:result=dict(success=False,reason=str(exc),state_known=False)
        results.append(result)
        emit(dict(event='step_finished',index=index,result=result))
        if result.get('success') is not True:
            return dict(success=False,completed_steps=index-1,failed_step=index,results=results)
    return dict(success=True,completed_steps=len(steps),results=results)
