"""Read-only learned grounding feasibility test on recorded manipulation RGB-D.

Only RGB and text enter the model. HSV regions are diagnostic references used
AFTER inference, never proposal boxes, crop hints, or model input corrections.
"""
import argparse
import json
import os
from pathlib import Path
import re
import sys
import time
import hashlib
import cv2
import numpy as np
import torch
from PIL import Image

LA=Path('/mnt/intel_ssd/locateanything')
ARM=Path('/home/aluupy/arm_feedback_experiments/20261027')
sys.path.insert(0,str(ARM/'src'));sys.path.insert(0,str(LA));sys.path.insert(0,str(LA/'Eagle/Embodied'))
from modules.feedback_observation import color_masks,mask_geometry
from eval_student import load_student,box_iou,padded_crop,roi_logits,paste_to_full,TEACHER
from distill.data import BOX_PAT,IM_END_ID

parser=argparse.ArgumentParser()
parser.add_argument('--out',required=True)
parser.add_argument('--model',choices=['prepare','prepare_controls','student06','student06_head','student06_head512','student17','teacher','teacher_native'],required=True)
parser.add_argument('--limit',type=int)
parser.add_argument('--bbox-only',action='store_true')
args=parser.parse_args()
out=Path(args.out)
out.mkdir(parents=True,exist_ok=True)

def regions(image):
    reference={}
    for color,mask,area in color_masks(image):
        if area<40:continue
        if color in reference and area<reference[color]['area']:continue
        ys,xs=np.where(mask)
        reference[color]=dict(box=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)],
                              center=[float(xs.mean()),float(ys.mean())],area=int(area),mask=mask>0)
    return reference

def make_dataset():
    manifest=[]
    sources=[('long_home','gripper_long_20261028_1','001_wrist.png'),
             ('long_refine','gripper_long_20261028_1','002_wrist.png'),
             ('long_lift','gripper_long_20261028_1','003_observer.png'),
             ('short_home','gripper_short_v3_20261028_1','001_wrist.png'),
             ('short_lift','gripper_short_v3_20261028_1','003_observer.png'),
             ('highres_home','feedback_camera_centered_01','002_wrist.png')]
    cn={'red':'红色方块','green':'绿色方块','blue':'蓝色方块'}
    for name,run,filename in sources:
        path=ARM/'runs'/run/filename
        image=cv2.imread(str(path));assert image is not None,path
        ref=regions(image);assert len(ref)>=2,(name,list(ref))
        W,H=image.shape[1],image.shape[0]
        for color,r in ref.items():
            np.save(out/f'{name}_{color}_reference.npy',r['mask'])
        def add(group,phrase,target,wrapper=True):
            prompt=f'Locate a single instance: {phrase}.' if wrapper else phrase
            manifest.append(dict(id=f'{name}_{len(manifest):03d}',image=str(path),frame=name,
                group=group,prompt=prompt,target=target,width=W,height=H,
                reference={c:{k:v for k,v in r.items() if k!='mask'} for c,r in ref.items()},
                rgb_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        for color in ['red','green','blue']:
            if color in ref:add('attribute_en',f'a {color} cube',color)
        order=sorted(ref,key=lambda c:ref[c]['center'][0])
        add('spatial_en','the leftmost cube in the image',order[0])
        if len(order)>1:
            anchor=order[-1];target=order[-2]
            add('relation_en',f'the cube immediately to the left of the {anchor} cube in the image',target)
        add('attribute_zh','红色方块','red')
        add('instruction_zh','请找出图中的红色方块，输出它的边界框。','red',False)
        add('absent_attribute','a yellow cube',None)
        if 'red' in ref and 'blue' in ref:
            relation='right' if ref['red']['center'][0]<ref['blue']['center'][0] else 'left'
            add('absent_relation',f'a red cube to the {relation} of the blue cube in the image',None)
        annotated=image.copy()
        for color,r in ref.items():
            x1,y1,x2,y2=r['box'];cv2.rectangle(annotated,(x1,y1),(x2,y2),(0,255,255),1)
            cv2.putText(annotated,color,(x1,max(12,y1-3)),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,255,255),1)
        cv2.imwrite(str(out/f'{name}_reference.jpg'),annotated)
    path=out/'manifest.json'
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    (out/'protocol.json').write_text(json.dumps(dict(
        references='HSV-derived regions of known colored cubes; diagnostic references, not manually annotated masks',
        model_inputs='Unmodified full RGB frame and text only',
        seed=42,reference_rules='Visible components >=40 pixels; left/right defined in image coordinates',
        all_inputs_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),cases=len(manifest)),indent=2))
    print('Prepared',len(manifest),'cases',flush=True)

if args.model=='prepare':
    make_dataset();sys.exit(0)
if args.model=='prepare_controls':
    from distill.data import DistillDataset
    from transformers import AutoProcessor
    processor=AutoProcessor.from_pretrained(str(LA/'LocateAnything-3B'),trust_remote_code=True,local_files_only=True)
    ds=DistillDataset([('/mnt/dataset/aluupy/annotations/refcocog_validation.jsonl',1,None)],
                      '/mnt/dataset/aluupy/images',processor)
    manifest=[]
    for i in range(8):
        fid,idx=ds.entries[ds.order[i]];line=json.loads(ds._lines[fid][idx])
        image_path=Path('/mnt/dataset/aluupy/images')/line['image']
        image=cv2.imread(str(image_path));H,W=image.shape[:2]
        match=BOX_PAT.search(line['conversations'][1]['value'])
        box=(np.asarray([int(v)/1000 for v in match.groups()])*[W,H,W,H]).astype(int).tolist()
        frame=f'control_{i:02d}'
        np.save(out/f'{frame}_object_reference.npy',np.load(line['mask_path']).astype(bool))
        manifest.append(dict(id=frame,image=str(image_path),frame=frame,group='in_domain_control',
            prompt=line['conversations'][0]['value'],target='object',width=W,height=H,
            reference=dict(object=dict(box=box)),rgb_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest()))
    path=out/'manifest.json'
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    print('Prepared 8 in-domain controls',flush=True);sys.exit(0)

torch.set_num_threads(6);torch.manual_seed(42)
from transformers import AutoProcessor,AutoModel
processor=AutoProcessor.from_pretrained(str(LA/'LocateAnything-3B'),trust_remote_code=True,local_files_only=True)
cases=json.loads((out/'manifest.json').read_text())
if args.limit:cases=cases[:args.limit]
model_dir=out/(args.model+'_bboxonly' if args.bbox_only else args.model)
model_dir.mkdir(exist_ok=False)
started=time.monotonic()
ckpt=(LA/'work_dirs/calib56K_plus_head/student_init.safetensors' if args.model.startswith('student06') else
      LA/'work_dirs/distill_full_qwen3_1p7_gtB/ckpt_last.pt')
teacher_path=str(LA/'LocateAnything-3B') if args.model=='teacher_native' else TEACHER
if args.model.startswith('teacher'):
    model=AutoModel.from_pretrained(teacher_path,torch_dtype=torch.bfloat16,trust_remote_code=True,local_files_only=True).to('cuda').eval()
    step=None
else:
    model,step=load_student(str(ckpt),'cuda')
box_head=None
head_path=None
if args.model.startswith('student06_head'):
    from distill.student import BoxRegHead
    head_hidden=512 if args.model=='student06_head512' else 128
    head_path=LA/('work_dirs/boxhead_calib56K_h512/ckpt_head.pt' if head_hidden==512 else
                 'work_dirs/boxhead_calib56K/ckpt_head.pt')
    box_head=BoxRegHead(model.llm.config.hidden_size,hidden=head_hidden).to('cuda')
    head_checkpoint=torch.load(head_path,map_location='cpu',weights_only=False)
    box_head.load_state_dict(head_checkpoint['box_head']);box_head.eval()
load_s=time.monotonic()-started
metadata=dict(model=args.model,checkpoint=teacher_path if args.model.startswith('teacher') else str(ckpt),step=step,
              box_head_checkpoint=None if head_path is None else str(head_path),
              box_head_hidden=None if head_path is None else head_hidden,
              outputs='bbox only' if args.bbox_only else 'bbox and student mask when available',
              load_s=load_s,gpu=torch.cuda.get_device_name(0),torch=torch.__version__,
              parameter_count=sum(p.numel() for p in model.parameters()),
              llm_hidden_size=None if args.model.startswith('teacher') else model.llm.config.hidden_size,
              note='student17 is an intermediate checkpoint' if args.model=='student17' else 'Read-only weights; no fine-tuning')
(model_dir/'metadata.json').write_text(json.dumps(metadata,indent=2))
print('MODEL_READY '+json.dumps(metadata),flush=True)

@torch.inference_mode()
def infer(image,prompt):
    torch.cuda.synchronize();start=time.monotonic()
    pil=Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    messages=[{'role':'user','content':[{'type':'image','image':pil},{'type':'text','text':prompt}]}]
    text=processor.py_apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    images,_=processor.process_vision_info(messages)
    inputs=processor(text=[text],images=images,return_tensors='pt')
    ids=inputs['input_ids'].to('cuda');pv=inputs['pixel_values'].to('cuda',torch.bfloat16)
    gh=torch.as_tensor(inputs['image_grid_hws'][0],dtype=torch.long,device='cuda')
    torch.cuda.synchronize();prepared=time.monotonic()
    vit=None
    if args.model.startswith('teacher'):
        result=model.generate(pixel_values=pv,input_ids=ids,attention_mask=torch.ones_like(ids),
            image_grid_hws=inputs['image_grid_hws'],tokenizer=processor.tokenizer,
            max_new_tokens=96,use_cache=True,generation_mode='hybrid' if args.model=='teacher_native' else 'slow',do_sample=False,temperature=1.,
            repetition_penalty=1.,verbose=False)
        answer=result[0] if isinstance(result,tuple) else result
        if not isinstance(answer,str):answer=processor.tokenizer.decode(answer[0],skip_special_tokens=False)
    else:
        embeds,vit=model.embed(pv,gh[None],ids)
        generated=model.llm.generate(inputs_embeds=embeds,attention_mask=torch.ones_like(ids),
            max_new_tokens=48,do_sample=False,eos_token_id=IM_END_ID)
        answer=processor.tokenizer.decode(generated[0],skip_special_tokens=False)
    boxes=[]
    for match in BOX_PAT.finditer(answer):
        box=[int(c)/1000 for c in match.groups()]
        if all(0<=v<=1 for v in box) and box[0]<box[2] and box[1]<box[3]:boxes.append(box)
    if box_head is not None:
        marker=processor.tokenizer.convert_tokens_to_ids('<box>')
        anchors=(generated[0]==marker).nonzero().flatten()
        boxes=[]
        if len(anchors):
            # Deployment-valid anchor from actual generated tokens, never a GT response.
            prefix=torch.cat([ids[0],generated[0][:int(anchors[0])+1]])
            full_embeds,_=model.embed(pv,gh[None],prefix[None])
            hidden=model.llm(inputs_embeds=full_embeds,attention_mask=torch.ones_like(prefix)[None],
                             output_hidden_states=True).hidden_states[-1][0,-1]
            box=box_head(hidden[None])[0].float().tolist()
            if all(0<=v<=1 for v in box) and box[0]<box[2] and box[1]<box[3]:boxes=[box]
    mask=None
    if len(boxes)==1 and vit is not None and not args.bbox_only:
        H,W=image.shape[:2]
        px=padded_crop(torch.tensor(boxes[0]),W,H)
        norm=torch.tensor([px[0]/W,px[1]/H,px[2]/W,px[3]/H])
        logits=roi_logits(vit,gh,norm,model.seg_head)
        mask=paste_to_full(torch.sigmoid(logits),px,W,H).cpu().numpy()
    torch.cuda.synchronize();end=time.monotonic()
    return dict(answer=answer,boxes=boxes,mask=mask,total_s=end-start,preprocess_s=prepared-start,network_s=end-prepared)

# One explicit warmup is excluded from the timed case distribution.
first=cases[0];infer(cv2.imread(first['image']),first['prompt'])
rows=[]
for case in cases:
    image=cv2.imread(case['image']);H,W=image.shape[:2]
    try:
        prediction=infer(image,case['prompt'])
        mask=prediction.pop('mask')
        boxes=prediction['boxes'];single=len(boxes)==1
        target=case['target'];refs=case['reference']
        ious={}
        if single:
            for color,r in refs.items():
                refbox=np.asarray(r['box'])/[W,H,W,H]
                ious[color]=float(box_iou(boxes[0],refbox))
        selected=max(ious,key=ious.get) if ious and max(ious.values())>0 else None
        target_iou=ious.get(target,0.) if target else None
        center_hit=False
        if target and single:
            reference=np.load(out/f'{case["frame"]}_{target}_reference.npy')
            cx=int((boxes[0][0]+boxes[0][2])*.5*W);cy=int((boxes[0][1]+boxes[0][3])*.5*H)
            center_hit=0<=cx<W and 0<=cy<H and bool(reference[cy,cx])
        mask_metrics={}
        if target and mask is not None:
            reference=np.load(out/f'{case["frame"]}_{target}_reference.npy')
            inter=int((mask&reference).sum());union=int((mask|reference).sum())
            mask_metrics=dict(mask_iou=inter/max(1,union),mask_dice=2*inter/max(1,int(mask.sum()+reference.sum())),
                              mask_pixels=int(mask.sum()))
        if mask is not None:
            np.save(model_dir/f'{case["id"]}_mask.npy',mask)
        explicit_refusal=not boxes and bool(re.search(r'not found|no (?:such|matching|yellow)|cannot find|不存在|未找到|没有',prediction['answer'],re.I))
        row=dict(**case,**prediction,selected=selected,target_iou=target_iou,
                 selection_correct=bool(target and selected==target),
                 localized=bool(target and target_iou>=.5),
                 center_hit=center_hit,
                 no_box=not boxes,explicit_refusal=explicit_refusal,**mask_metrics)
        vis=image.copy()
        if mask is not None:
            vis[mask]=(vis[mask]*.6+np.array([255,80,0])*.4).astype(np.uint8)
        for box in boxes:
            x1,y1,x2,y2=(np.asarray(box)*[W,H,W,H]).astype(int)
            cv2.rectangle(vis,(x1,y1),(x2,y2),(0,0,255),2)
        if target:
            x1,y1,x2,y2=refs[target]['box'];cv2.rectangle(vis,(x1,y1),(x2,y2),(0,255,0),1)
        cv2.imwrite(str(model_dir/f'{case["id"]}.jpg'),vis)
    except Exception as exc:
        import traceback;traceback.print_exc()
        row=dict(**case,error=repr(exc),localized=False,selection_correct=False)
    rows.append(row)
    with (model_dir/'cases.jsonl').open('a') as stream:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    print(json.dumps(dict(model=args.model,id=case['id'],group=case['group'],target=case['target'],
        selected=row.get('selected'),iou=row.get('target_iou'),seconds=row.get('total_s'),error=row.get('error'))),flush=True)
(model_dir/'results.json').write_text(json.dumps(dict(metadata=metadata,cases=rows),indent=2,ensure_ascii=False))
print('DONE '+args.model,flush=True)
