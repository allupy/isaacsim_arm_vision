"""Summaries computed from raw predictions; references never feed the models."""
import json
from pathlib import Path
import sys
import numpy as np
root=Path(sys.argv[1]);summaries=[]
for dataset,directory in [('scene',root),('controls',root/'controls')]:
    for model_dir in sorted(directory.iterdir()):
        path=model_dir/'results.json'
        if not path.exists():continue
        result=json.loads(path.read_text());rows=result['cases']
        groups={}
        for group in sorted({r['group'] for r in rows}):
            part=[r for r in rows if r['group']==group]
            positive=[r for r in part if r['target'] is not None]
            negative=[r for r in part if r['target'] is None]
            groups[group]=dict(n=len(part),localized=sum(bool(r.get('localized')) for r in positive),
                no_box=sum(bool(r.get('no_box')) for r in negative),
                explicit_refusal=sum(bool(r.get('explicit_refusal')) for r in negative))
        positive=[r for r in rows if r['target'] is not None]
        negative=[r for r in rows if r['target'] is None]
        hits=0
        for row in positive:
            boxes=row.get('boxes',[])
            if len(boxes)!=1:continue
            box=boxes[0];W,H=row['width'],row['height']
            cx=int((box[0]+box[2])*.5*W);cy=int((box[1]+box[3])*.5*H)
            reference=np.load(directory/f'{row["frame"]}_{row["target"]}_reference.npy')
            hits+=int(0<=cx<W and 0<=cy<H and reference[cy,cx])
        timings=[r['total_s'] for r in rows if 'total_s' in r]
        summary=dict(dataset=dataset,model=model_dir.name,metadata=result['metadata'],n=len(rows),
            errors=sum('error' in r for r in rows),positive_n=len(positive),
            localized=sum(bool(r.get('localized')) for r in positive),center_hits=hits,
            mean_box_iou=float(np.mean([r.get('target_iou') or 0. for r in positive])),
            mean_mask_iou=None if model_dir.name.startswith('teacher') else float(np.mean([r.get('mask_iou',0.) for r in positive])),
            negative_n=len(negative),negative_no_box=sum(bool(r.get('no_box')) for r in negative),
            explicit_refusals=sum(bool(r.get('explicit_refusal')) for r in negative),
            latency_median_s=float(np.median(timings)) if timings else None,
            latency_p95_s=float(np.percentile(timings,95)) if timings else None,groups=groups)
        summaries.append(summary)
(root/'summary.json').write_text(json.dumps(summaries,indent=2))
for r in summaries:
    print(json.dumps({k:v for k,v in r.items() if k not in ['metadata','groups']}))
