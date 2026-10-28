"""Encode captured rendered frames using their actual simulation/wall timestamps."""
import argparse
import bisect
import json
import math
from pathlib import Path
import subprocess
import cv2
import numpy as np
import imageio_ffmpeg

parser=argparse.ArgumentParser()
parser.add_argument('--root',default='/experiment')
parser.add_argument('--prefix',default='record20261028')
parser.add_argument('--count',type=int,default=3)
args=parser.parse_args()
root=Path(args.root)
destination=root/'videos/encoded'
destination.mkdir(exist_ok=True,parents=True)
manifest=[]
fps=30

for take in range(1,args.count+1):
    identifier=f'{args.prefix}_{take}'
    source=root/'videos'/identifier
    frames=json.loads((source/'frames.json').read_text())
    capture=json.loads((source/'capture.json').read_text())
    run=root/'runs'/identifier
    result=json.loads((run/'result.json').read_text())
    gripper=result.get('gripper',{}).get('label','ORIGINAL ROBOTIQ 2F-140')
    events=[json.loads(s) for s in (run/'events.jsonl').read_text().splitlines()]
    events=[e for e in events if 'state' in e and 'sim_time' in e]
    if not frames or not events:
        raise ValueError('Recording or state metadata missing')
    variants=['sim_1x']+(['wall_1x'] if take==1 else [])
    for variant in variants:
        axis='sim_time' if variant=='sim_1x' else 'wall_time'
        event_axis='sim_time' if variant=='sim_1x' else 'time'
        frame_times=[f[axis] for f in frames]
        event_times=[e[event_axis] for e in events]
        start=max(frame_times[0],event_times[0]-.6)
        stop=min(frame_times[-1],event_times[-1]+1.)
        count=math.ceil((stop-start)*fps)
        output=destination/f'{identifier}_{variant}.mp4'
        if output.exists():
            raise FileExistsError(output)
        command=[imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-f','rawvideo',
            '-vcodec','rawvideo','-pix_fmt','bgr24','-s','1280x480','-r',str(fps),'-i','-',
            '-an','-c:v','libx264','-preset','fast','-crf','20','-threads','2','-pix_fmt','yuv420p',
            '-movflags','+faststart',str(output)]
        encoder=subprocess.Popen(command,stdin=subprocess.PIPE)
        cached_index=None;cached_pair=None;sheet=[]
        checkpoints=set(np.linspace(0,count-1,9,dtype=int).tolist())
        try:
            for index in range(count):
                timestamp=start+index/fps
                fi=max(0,bisect.bisect_right(frame_times,timestamp)-1)
                metadata=frames[fi]
                if fi!=cached_index:
                    observer=cv2.imread(str(source/f'{metadata["index"]:06d}_observer.jpg'))
                    wrist=cv2.imread(str(source/f'{metadata["index"]:06d}_wrist.jpg'))
                    if observer is None or wrist is None:
                        raise IOError('Captured frame missing')
                    cached_pair=np.hstack([cv2.resize(observer,(640,360)),cv2.resize(wrist,(640,360))])
                    cached_index=fi
                canvas=np.full((480,1280,3),(25,24,20),np.uint8)
                canvas[80:440]=cached_pair
                state=events[max(0,bisect.bisect_right(event_times,timestamp)-1)]['state']
                sim_elapsed=metadata['sim_time']-events[0]['sim_time']
                wall_elapsed=metadata['wall_time']-events[0]['time']
                cv2.putText(canvas,f'{gripper} | Take {take} | HSV color | YOLO-World: OFF',
                            (18,25),cv2.FONT_HERSHEY_SIMPLEX,.62,(240,240,240),1,cv2.LINE_AA)
                cv2.putText(canvas,f'{state}  |  Simulation: {sim_elapsed:5.1f}s  |  Recorded wall time: {wall_elapsed:5.1f}s',
                            (18,54),cv2.FONT_HERSHEY_SIMPLEX,.65,(90,220,250),1,cv2.LINE_AA)
                cv2.putText(canvas,'EXTERNAL CAMERA',(18,75),cv2.FONT_HERSHEY_SIMPLEX,.45,(190,190,190),1)
                cv2.putText(canvas,'WRIST CAMERA',(658,75),cv2.FONT_HERSHEY_SIMPLEX,.45,(190,190,190),1)
                label='SIMULATION TIME 1x' if variant=='sim_1x' else 'RECORDED WALL TIME 1x (includes recording overhead)'
                cv2.putText(canvas,label+' | Actual rendered frames; no synthetic motion',
                            (18,463),cv2.FONT_HERSHEY_SIMPLEX,.52,(220,220,220),1,cv2.LINE_AA)
                if state in ['SUCCESS','FAILED']:
                    evaluation=result.get('ground_truth_evaluation',{})
                    score=evaluation.get('placement_xy_error')
                    if score is not None:
                        text=f'Independent final XY error: {score*1000:.1f} mm | pass: {evaluation.get("task_success")}'
                        cv2.putText(canvas,text,(20,427),cv2.FONT_HERSHEY_SIMPLEX,.55,(30,255,255),2,cv2.LINE_AA)
                encoder.stdin.write(canvas.tobytes())
                if index in checkpoints:
                    sheet.append(cv2.resize(canvas,(640,240)))
                if index==min(fps,count-1):
                    cv2.imwrite(str(output.with_suffix('.jpg')),canvas)
        finally:
            encoder.stdin.close()
            if encoder.wait()!=0:
                raise RuntimeError('Video encoder failed')
        # Decode every frame to verify the final H.264 artifact, not just the input JPEGs.
        reader=cv2.VideoCapture(str(output));decoded=0
        while True:
            ok,frame=reader.read()
            if not ok:break
            if frame.shape[:2]!=(480,1280):raise ValueError('Bad video dimensions')
            decoded+=1
        reader.release()
        if decoded!=count:
            raise ValueError(f'Video decode count mismatch: {decoded} vs {count}')
        while len(sheet)<9:sheet.append(sheet[-1])
        contact=np.vstack([np.hstack(sheet[i:i+3]) for i in range(0,9,3)])
        cv2.imwrite(str(output.with_name(output.stem+'_contact.jpg')),contact)
        row=dict(id=identifier,variant=variant,path=str(output),frames=count,fps=fps,duration_s=count/fps,
                 gripper=result.get('gripper',{}),
                 captured_frames=len(frames),dropped_frames=capture['dropped_frames'],perception='color',
                 policy_success=result.get('success'),evaluation=result.get('ground_truth_evaluation'),
                 recording_policy_wall_s=result.get('elapsed_s'))
        manifest.append(row)
        (destination/f'{args.prefix}_manifest.json').write_text(json.dumps(manifest,indent=2))
        print(json.dumps(row),flush=True)
