"""Plot measured simulator object traces; no synthesized trajectory data."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

root=Path(__file__).resolve().parents[1]/'results'/'diagnosis'
run=root/'runs/feedback_color_05'
result=json.loads((run/'result.json').read_text())
truth=[json.loads(line) for line in (run/'ground_truth.jsonl').read_text().splitlines()]
target=result['ground_truth_evaluation']['target']
xyz=np.array([row['objects'][target] for row in truth])
t=np.array([row['sim_time'] for row in truth]);t-=t[0]
goal=np.array(result['config']['place_surface'][:2])
figure,axes=plt.subplots(1,2,figsize=(11,4.2),constrained_layout=True)
axes[0].plot(t,(xyz[:,2]-xyz[0,2])*100,color='#d94b49',lw=2)
axes[0].axhline(result['config']['min_lift']*100,color='#777777',ls='--',label='Lift verification threshold')
axes[0].set(xlabel='Simulation time since recording start (s)',ylabel='Target height above initial support (cm)',
            title='Measured target lift')
axes[0].grid(alpha=.2);axes[0].legend(fontsize=8)
axes[1].plot(xyz[:,0],xyz[:,1],color='#d94b49',lw=2,label='Measured target path')
axes[1].scatter(*xyz[0,:2],s=55,marker='o',color='#333333',label='Start')
axes[1].scatter(*xyz[-1,:2],s=65,marker='s',color='#d94b49',label='Final')
axes[1].scatter(*goal,s=80,marker='x',color='#225b99',label='Requested placement')
axes[1].add_patch(Circle(goal,result['config']['place_tolerance_xy'],facecolor='#d8e7f5',edgecolor='#225b99',alpha=.4))
axes[1].set(xlabel='World x (m)',ylabel='World y (m)',title='Placement within the configured 6 cm region',aspect='equal')
axes[1].grid(alpha=.2);axes[1].legend(fontsize=8,loc='upper left')
error=result['ground_truth_evaluation']['placement_xy_error']*1000
figure.suptitle(f'Isaac Sim diagnostic: one complete pick-and-place | final XY error {error:.1f} mm',fontsize=12)
out=root/'figures';out.mkdir(exist_ok=True)
figure.savefig(out/'closed_loop_evidence.png',dpi=170)
print(out/'closed_loop_evidence.png')
