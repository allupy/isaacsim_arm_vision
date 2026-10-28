"""Local Chinese-captioned continuous recording, using actual frame timestamps."""
import argparse,bisect,json,math,subprocess
from pathlib import Path
import cv2,numpy as np,imageio_ffmpeg
from PIL import Image,ImageDraw,ImageFont
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--run-id',default='agent_llm_video_20261028');args=p.parse_args()
root=Path(args.root);identifier=args.run_id;source=root/'videos'/identifier;agent=root/'agent_runs'/identifier
read=lambda path:json.loads(path.read_text(encoding='utf-8'))
frames=read(source/'frames.json');plan=read(agent/'plan.json');result=read(agent/'result.json');audit=read(agent/'audit.json')
physical=any(s['destination'] in ['plate','cup_stand'] for s in plan['steps'])
names={'apple':'苹果','cup':'杯子','fruit_zone':'水果区','cup_zone':'杯子区','plate':'盘子','cup_stand':'杯架'}
step_labels=[names[s['target']]+' → '+names[s['destination']] for s in plan['steps']]
title='，再'.join('把'+names[s['target']]+'放到'+names[s['destination']] for s in plan['steps'])
events=[json.loads(s) for s in (agent/'events.jsonl').read_text().splitlines()]
states={}
for i in range(1,len(plan['steps'])+1):
    states[i]=[json.loads(s) for s in (root/'runs'/f'{identifier}_{i}'/'events.jsonl').read_text().splitlines()]
    states[i]=[e for e in states[i] if 'state' in e]
fonts={size:ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',size) for size in [18,22,26,32]}
labels={'HOME':'准备','ACQUIRE':'寻找目标','APPROACH':'接近对准','REFINE':'近距离校正','DESCEND':'接近目标','CLOSE':'夹持',
        'LIFT':'提起','VERIFY_LIFT':'确认抓取','TRANSFER':'搬运','PLACE':'放置','RELEASE':'松开','RETREAT':'撤离','VERIFY_PLACE':'确认放置','SUCCESS':'完成','FAILED':'失败','RECOVER':'恢复检查'}
out=root/'videos/encoded';out.mkdir(parents=True,exist_ok=True);output=out/'llm_tabletop_agent_cn.mp4'
if output.exists():raise FileExistsError(output)
times=[f['sim_time'] for f in frames];start=times[0];end=times[-1];count=math.ceil((end-start)*30)
cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r','30','-i','-',
     '-an','-c:v','libx264','-preset','fast','-crf','20','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(output)]
encoder=subprocess.Popen(cmd,stdin=subprocess.PIPE);cached=-1;sheet=[];checkpoints=set(np.linspace(0,count-1,9,dtype=int))
try:
    for index in range(count):
        fi=max(0,bisect.bisect_right(times,start+index/30)-1);frame=frames[fi];wall=frame['wall_time']
        if fi!=cached:
            observer=cv2.imread(str(source/f'{frame["index"]:06d}_observer.jpg'));wrist=cv2.imread(str(source/f'{frame["index"]:06d}_wrist.jpg'))
            if observer is None or wrist is None:raise IOError('Missing source frame')
            scene=Image.fromarray(cv2.cvtColor(cv2.resize(observer,(960,540)),cv2.COLOR_BGR2RGB))
            hand=Image.fromarray(cv2.cvtColor(cv2.resize(wrist,(320,180)),cv2.COLOR_BGR2RGB));cached=fi
        canvas=Image.new('RGB',(1280,720),(18,25,35));canvas.paste(scene,(0,130));canvas.paste(hand,(960,130))
        draw=ImageDraw.Draw(canvas)
        draw.text((22,15),title,font=fonts[32],fill='white')
        model=(plan.get('llm_response') or {}).get('model','unavailable')
        draw.text((24,64),f'LLM: {model}  →  计划校验  →  抓放技能  →  结果反馈',font=fonts[22],fill=(140,205,245))
        draw.text((24,103),'机械臂全景',font=fonts[18],fill='white');draw.text((980,103),'腕部相机',font=fonts[18],fill='white')
        begun=[e for e in events if e['event']=='step_started' and e['time']<=wall]
        current=begun[-1]['index'] if begun else 1
        finished={e['index']:e['result']['success'] for e in events if e['event']=='step_finished' and e['time']<=wall}
        stage=[e for e in states[current] if e['time']<=wall];stage=labels.get(stage[-1]['state'],'执行中') if stage else '准备视觉模型'
        draw.text((980,330),'执行计划',font=fonts[26],fill='white')
        for i,text in enumerate(step_labels,1):
            mark='完成：' if finished.get(i) else ('当前：' if i==current else '待办：')
            color=(110,230,170) if finished.get(i) else 'white'
            draw.text((980,375+(i-1)*47),mark+text,font=fonts[22],fill=color)
        done=len(finished)==2 and all(finished.values()) and audit['verified_success']
        scored=len(finished)==2 and all(finished.values())
        status='任务完成' if done else ('位置验证未通过' if scored else stage)
        draw.text((980,490),status,font=fonts[26],fill=(110,230,170) if done else (255,211,105))
        if scored:
            for i,score in enumerate(audit['steps']):
                draw.text((980,540+i*32),f'{["苹果","杯子"][i]}误差 {score["placement_xy_error"]*1000:.1f} mm',font=fonts[18],fill='white')
        mode='承托目标使用视觉标记定位' if physical else '固定放置区域'
        draw.text((22,682),f'真实仿真连续录制 · {mode} · 仿真时间 1 倍速   {index/30:.1f}s',font=fonts[22],fill='white')
        encoder.stdin.write(canvas.tobytes())
        if index in checkpoints:sheet.append(np.asarray(canvas.resize((640,360))))
        if index==min(count-1,30):canvas.save(out/'llm_tabletop_agent.jpg')
finally:
    encoder.stdin.close()
    if encoder.wait()!=0:raise RuntimeError('Encoder failed')
cap=cv2.VideoCapture(str(output));decoded=0
while True:
    ok,img=cap.read()
    if not ok:break
    assert img.shape[:2]==(720,1280);decoded+=1
cap.release();assert decoded==count
Image.fromarray(np.vstack([np.hstack(sheet[i:i+3]) for i in range(0,9,3)])).save(out/'contact.jpg')
(out/'manifest.json').write_text(json.dumps(dict(path=str(output),frames=count,duration_s=count/30,capture=read(source/'capture.json'),audit=audit),indent=2))
print('Verified',count,'frames;',count/30,'seconds',flush=True)
