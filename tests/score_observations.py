"""Offline only: compare saved detections to simultaneous simulator ground truth."""
import argparse
import json
from pathlib import Path
import numpy as np


def score(run):
    truth = {}
    for line in (run/'ground_truth.jsonl').read_text().splitlines():
        row = json.loads(line)
        truth[row['stamp_ns']] = row
    rows = []
    for line in (run/'events.jsonl').read_text().splitlines():
        event = json.loads(line)
        if event.get('kind') != 'observation' or event['stamp_ns'] not in truth:
            continue
        frame_truth = truth[event['stamp_ns']]['objects']
        for obs in event['candidates']:
            name = obs['label']+'_cube'
            if name not in frame_truth:
                continue  # no invented matching or class identity for open-vocabulary detections
            position = np.array(obs['position'])
            center = np.array(frame_truth[name])
            top = center+np.array([0.,0.,.025])
            rows.append(dict(camera=event['camera'], target=name, stamp_ns=event['stamp_ns'],
                             xy_error_m=float(np.linalg.norm((position-center)[:2])),
                             surface_z_error_m=float(abs(position[2]-top[2]))))
    return dict(run=str(run), samples=rows,
                note='Calibrated simulator RGB-D; color segmentation diagnostic; not an open-vocabulary or grasp success benchmark.',
                summary={camera:dict(n=len(part), xy_mean_m=float(np.mean([r['xy_error_m'] for r in part])),
                                     xy_max_m=float(max(r['xy_error_m'] for r in part)),
                                     surface_z_mean_m=float(np.mean([r['surface_z_error_m'] for r in part])))
                         for camera in ('wrist','observer')
                         if (part := [r for r in rows if r['camera'] == camera])})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    result = score(args.run)
    (args.run/'perception_score.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
