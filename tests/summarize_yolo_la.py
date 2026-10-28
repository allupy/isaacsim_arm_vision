import json
from pathlib import Path
import sys
import numpy as np
root=Path(sys.argv[1]);rows=[]
yolo=json.loads((root/'yolo_world/bbox_results.json').read_text())
for size in [640,1280]:
    for conf in [.02,.25]:
        cases=[c for c in yolo['rows'] if c['imgsz']==size and c['conf']==conf]
        positive=[c for c in cases if c['target']];negative=[c for c in cases if not c['target']]
        timings=[t for entry in yolo['timings'] if entry['imgsz']==size for t in entry['samples']]
        rows.append(dict(model='YOLO-World',imgsz=size,conf=conf,positive_n=len(positive),
            localized=sum(c['localized'] for c in positive),negative_n=len(negative),
            correct_absence=sum(c['correct_absence'] for c in negative),
            mean_box_iou=float(np.mean([c['target_iou'] for c in positive])),
            median_s=float(np.median(timings)),p95_s=float(np.percentile(timings,95)),
            latency_unit='One frame pass detects all four cached categories'))
sam_path=root/'yolo_world/sam_results.json'
if sam_path.exists():
    sam=json.loads(sam_path.read_text())['rows']
    positives=[c for c in sam if c['target']]
    rows.append(dict(model='YOLO-World 1280 + SAM ViT-B',detected_cases=len(sam),
        positive_detected=len(positives),total_positive=17,
        detected_median_s=float(np.median([c['estimated_yolo_sam_s'] for c in sam])),
        mean_mask_iou_detected=float(np.mean([c['mask_iou'] for c in positives])),
        mean_mask_iou_all=sum(c['mask_iou'] for c in positives)/17,
        note='Sum of separately timed YOLO and fresh SAM encoding/decoding medians; only detected boxes invoke SAM'))
for name in ['student06_bboxonly','teacher_native_bboxonly']:
    path=root/name/'results.json'
    if not path.exists():continue
    result=json.loads(path.read_text());cases=result['cases']
    positive=[c for c in cases if c['target']];negative=[c for c in cases if not c['target']]
    timings=[c['total_s'] for c in cases]
    rows.append(dict(model=name,positive_n=len(positive),localized=sum(c['localized'] for c in positive),
        negative_n=len(negative),correct_absence=sum(c['no_box'] for c in negative),
        mean_box_iou=float(np.mean([c['target_iou'] or 0 for c in positive])),
        median_s=float(np.median(timings)),p95_s=float(np.percentile(timings,95)),
        latency_unit='One RGB + single-target text query, bbox only',
        metadata=result['metadata']))
(root/'summary.json').write_text(json.dumps(rows,indent=2))
for row in rows:print(json.dumps({k:v for k,v in row.items() if k!='metadata'}))
