"""Compare custom-image input tensors with the project's established eval path."""
import json
from pathlib import Path
import sys
sys.path.insert(0,'/mnt/intel_ssd/locateanything')
sys.path.insert(0,'/mnt/intel_ssd/locateanything/Eagle/Embodied')
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor
from distill.data import DistillDataset
root=Path(sys.argv[1])
case=json.loads((root/'manifest.json').read_text())[0]
processor=AutoProcessor.from_pretrained('/mnt/dataset/aluupy/LocateAnything-3B',trust_remote_code=True,local_files_only=True)
sample=dict(image=case['image'],conversations=[dict(from_='human',value=case['prompt']),
    dict(from_='gpt',value='<ref>placeholder</ref><box><100><100><900><900></box>')])
# Dataset reads values only; the placeholder response is sliced out before comparison.
path=root/'parity_sample.jsonl';path.write_text(json.dumps(sample)+'\n')
dataset=DistillDataset([(str(path),1,None)],'',processor);d=dataset[0]
image=Image.open(case['image']).convert('RGB')
messages=[dict(role='user',content=[dict(type='image',image=image),dict(type='text',text=case['prompt'])])]
text=processor.py_apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
images,_=processor.process_vision_info(messages)
inputs=processor(text=[text],images=images,return_tensors='pt')
ids=d['input_ids'][:d['resp_start']]
result=dict(prompt_ids_equal=torch.equal(ids,inputs['input_ids'][0]),
            dataset_prompt_tokens=len(ids),test_prompt_tokens=len(inputs['input_ids'][0]),
            pixels_equal=torch.equal(d['pixel_values'],inputs['pixel_values']),
            grid_equal=np.array_equal(d['grid_hws'].numpy(),inputs['image_grid_hws'][0]),
            dataset_suffix=processor.tokenizer.decode(ids[-12:],skip_special_tokens=False),
            test_suffix=processor.tokenizer.decode(inputs['input_ids'][0,-12:],skip_special_tokens=False))
(root/'input_parity.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
assert result['prompt_ids_equal'] and result['pixels_equal'] and result['grid_equal']
