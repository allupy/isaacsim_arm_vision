import hashlib
import json
from pathlib import Path
import shutil
import sys
source=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=False)
cases=[c for c in json.loads((source/'manifest.json').read_text()) if c['group'] in ['attribute_en','absent_attribute']]
(out/'manifest.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2))
for path in source.glob('*_reference.npy'):shutil.copyfile(path,out/path.name)
(out/'protocol.json').write_text(json.dumps(dict(
    source_manifest_sha256=hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest(),
    cases=len(cases),positive=sum(c['target'] is not None for c in cases),
    scope='Same 6 full RGB frames; English color categories plus absent yellow; not free-form relational reasoning',
    yolo_vocabulary=['red cube','green cube','blue cube','yellow cube'],
    primary_yolo_conf=.02,secondary_yolo_conf=.25,reference='HSV regions for scoring only',
    latency='Warm GPU-synchronized wall time, excludes model/text setup and all scoring/disk I/O',
    sampling='YOLO: one multi-class frame pass, 5 timed repeats/frame; LA: one single-target query/pass'),indent=2))
print('Prepared',len(cases),'matched attribute requests')
