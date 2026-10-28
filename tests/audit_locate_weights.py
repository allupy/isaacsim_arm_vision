"""Checkpoint identity audit without assembling or modifying neural models."""
import json
from pathlib import Path
import sys
import torch
from safetensors import safe_open
root=Path('/mnt/intel_ssd/locateanything/work_dirs')
results={}
with safe_open(root/'calib56K_plus_head/student_init.safetensors',framework='pt',device='cpu') as file:
    keys=list(file.keys())
    results['student06']=dict(
        shapes={k:file.get_slice(k).get_shape() for k in keys if k.endswith('embed_tokens.weight')},
        seg_head_keys=sum('seg_head.' in k for k in keys))
checkpoint=torch.load(root/'distill_full_qwen3_1p7_gtB/ckpt_last.pt',map_location='cpu',weights_only=False,mmap=True)
state=checkpoint['student']
results['student17']=dict(step=checkpoint.get('opt_step'),
    shapes={k:list(v.shape) for k,v in state.items() if k.endswith('embed_tokens.weight')},
    seg_head_keys=sum('seg_head.' in k for k in state))
out=Path(sys.argv[1]);out.write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
