import json,sys
from pathlib import Path
p=Path('/experiment/runs')/sys.argv[1]
result=json.loads((p/'result.json').read_text());print('RESULT',result['success'],result['reason'])
for e in map(json.loads,(p/'events.jsonl').read_text().splitlines()):
    if e.get('kind')=='observation':
        print(e['camera'],[(x['label'],round(x['confidence'],3),[round(v,3) for v in x['position']],
                            [round(v,2) for v in x['appearance']],x.get('source')) for x in e['candidates']])
    if e.get('state') in ['VERIFY_LIFT','RECOVER','FAILED']:print(e)
