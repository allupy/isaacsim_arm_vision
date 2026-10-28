import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(exist_ok=True,parents=True)
rows=json.loads((root/'performance_summary.json').read_text())
groups={label:[r for r in rows if r['run'].startswith(prefix)] for label,prefix in
        [('Baseline','performance_baseline_'),('Optimized','performance_production_')]}
aggregate={}
for label,data in groups.items():
    aggregate[label]=dict(trials=len(data),policy_successes=sum(bool(r['success']) for r in data),
        truth_successes=sum(bool(r['truth_success']) for r in data),
        wall_s=[r['elapsed_s'] for r in data],sim_s=[r.get('elapsed_sim_s') for r in data],
        placement_mm=[r['placement_error_m']*1000 for r in data],
        median_wall_s=float(np.median([r['elapsed_s'] for r in data])),
        median_placement_mm=float(np.median([r['placement_error_m']*1000 for r in data])),
        median_command_gap_p95_ms=float(np.median([r['moving_command_gap_sim_s_p95']*1000 for r in data])),
        max_arm_command_step_rad=max(r['moving_command_max_step_rad'] for r in data))
(out/'aggregate.json').write_text(json.dumps(aggregate,indent=2))
fig,axes=plt.subplots(1,3,figsize=(13,3.8))
colors=['#bc6262','#267c98']
for i,(label,data) in enumerate(aggregate.items()):
    axes[0].bar(i,data['median_wall_s'],color=colors[i],alpha=.7)
    axes[0].scatter(np.full(len(data['wall_s']),i),data['wall_s'],color='black',s=17,zorder=4)
    axes[1].scatter(np.full(len(data['placement_mm']),i),data['placement_mm'],color=colors[i],s=45)
axes[0].set(title='Complete cycle (wall time)',ylabel='Seconds',xticks=[0,1],xticklabels=list(aggregate))
axes[1].axhline(60,color='gray',ls='--',label='Scoring tolerance: 60 mm')
axes[1].set(title='Independent final placement error',ylabel='Millimeters',xticks=[0,1],xticklabels=list(aggregate))
axes[1].legend(fontsize=8)
for label,run,color in [('Baseline','performance_baseline_3',colors[0]),
                      ('Optimized','performance_production_3',colors[1])]:
    directory=root/run
    events=[json.loads(s) for s in (directory/'events.jsonl').read_text().splitlines()]
    start=next(e for e in events if e.get('state')=='PLACE')
    finish=next(e for e in events if e.get('kind')=='motion' and e['time']>start['time'])
    commands=[json.loads(s) for s in (directory/'command_updates.jsonl').read_text().splitlines()]
    commands=[c for c in commands if start['sim_time']<=c[0]<=finish['sim_time'] and 'elbow_joint' in c[2]]
    x=np.array([c[0] for c in commands]); y=np.array([c[3][c[2].index('elbow_joint')] for c in commands])
    axes[2].plot(x-x[0],np.rad2deg(y-y[0]),'.-',ms=2.5,lw=1.,color=color,label=label)
axes[2].set(title='Place descent: commanded elbow angle',xlabel='Simulation seconds',ylabel='Change (degrees)')
axes[2].legend(fontsize=8)
for ax in axes:
    ax.grid(alpha=.2,axis='y');ax.spines[['top','right']].set_visible(False)
fig.suptitle('Same scene reset before each trial; color diagnostic perception; n=3 per profile',fontsize=11)
fig.tight_layout()
fig.savefig(out/'performance_comparison.png',dpi=170)
print(json.dumps(aggregate,indent=2))
