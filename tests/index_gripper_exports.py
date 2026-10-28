"""Build a combined export index after independently encoded gripper groups."""
import json
from pathlib import Path
import sys
import cv2
root=Path(sys.argv[1]);directory=root/'videos/encoded'
rows=[]
for prefix in ['gripper_long_20261028','gripper_short_v3_20261028']:
    for take in range(1,4):
        identifier=f'{prefix}_{take}'
        result=json.loads((root/'runs'/identifier/'result.json').read_text())
        for variant in ['sim_1x','wall_1x']:
            path=directory/f'{identifier}_{variant}.mp4'
            if not path.exists():continue
            reader=cv2.VideoCapture(str(path))
            count=int(reader.get(cv2.CAP_PROP_FRAME_COUNT));fps=reader.get(cv2.CAP_PROP_FPS)
            reader.release()
            rows.append(dict(id=identifier,variant=variant,path=str(path),frames=count,fps=fps,
                duration_s=count/fps,gripper=result['gripper'],
                evaluation=result['ground_truth_evaluation'],perception=result['perception']))
(directory/'manifest.json').write_text(json.dumps(rows,indent=2))
print(f'Indexed {len(rows)} encoded grasp videos')
