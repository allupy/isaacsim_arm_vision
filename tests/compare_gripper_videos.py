"""Stack matched videos at unchanged 1x simulation speed, aligned to HOME."""
import argparse
import json
from pathlib import Path
import subprocess
import cv2
import numpy as np
import imageio_ffmpeg
parser=argparse.ArgumentParser()
parser.add_argument('--root',required=True)
parser.add_argument('--take',type=int,default=1)
parser.add_argument('--short-prefix',default='gripper_short_v3_20261028')
args=parser.parse_args()
directory=Path(args.root)/'videos/encoded'
paths=[directory/f'gripper_long_20261028_{args.take}_sim_1x.mp4',
       directory/f'{args.short_prefix}_{args.take}_sim_1x.mp4']
readers=[cv2.VideoCapture(str(path)) for path in paths]
counts=[int(r.get(cv2.CAP_PROP_FRAME_COUNT)) for r in readers]
assert all(n>0 for n in counts)
output=directory/f'gripper_comparison_{args.take}_sim_1x.mp4'
if output.exists():raise FileExistsError(output)
process=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error',
    '-f','rawvideo','-vcodec','rawvideo','-pix_fmt','bgr24','-s','1280x960','-r','30','-i','-',
    '-an','-c:v','libx264','-preset','fast','-crf','20','-threads','2','-pix_fmt','yuv420p',
    '-movflags','+faststart',str(output)],stdin=subprocess.PIPE)
last=[None,None];screens=[]
try:
    for index in range(max(counts)):
        parts=[]
        for i,reader in enumerate(readers):
            if index<counts[i]:
                ok,frame=reader.read()
                if not ok:raise IOError(paths[i])
                last[i]=frame
            frame=last[i].copy()
            if index>=counts[i]:
                cv2.putText(frame,'Completed: holding final recorded frame',(620,463),
                            cv2.FONT_HERSHEY_SIMPLEX,.5,(0,230,255),1,cv2.LINE_AA)
            parts.append(frame)
        combined=np.vstack(parts)
        process.stdin.write(combined.tobytes())
        if index in [30,max(counts)//3,2*max(counts)//3,max(counts)-1]:screens.append(combined.copy())
finally:
    process.stdin.close()
    if process.wait()!=0:raise RuntimeError('ffmpeg failed')
    for reader in readers:reader.release()
reader=cv2.VideoCapture(str(output));decoded=0
while True:
    ok,frame=reader.read()
    if not ok:break
    assert frame.shape==(960,1280,3)
    decoded+=1
reader.release()
assert decoded==max(counts)
cv2.imwrite(str(output.with_suffix('.jpg')),screens[1])
cv2.imwrite(str(output.with_name(output.stem+'_contact.jpg')),np.hstack([cv2.resize(f,(640,480)) for f in screens]))
data=dict(output=str(output),source_videos=[str(p) for p in paths],fps=30,frames=decoded,
          duration_s=decoded/30,alignment='HOME start, simulation time 1x; final frame held after completion')
output.with_suffix('.json').write_text(json.dumps(data,indent=2))
print(json.dumps(data))
