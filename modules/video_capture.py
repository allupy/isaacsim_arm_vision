"""Optional capture of real rendered simulator frames; no policy inputs."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import time
import cv2
import numpy as np


class ExperimentVideoCapture:
    def __init__(self, sim_directory, video_directory):
        self.sim_directory=Path(sim_directory)
        self.video_directory=Path(video_directory)
        self.active=False
        self.writer=ThreadPoolExecutor(max_workers=1,thread_name_prefix='video-jpeg')
        self.pending=[]

    def requests(self):
        request=self.sim_directory/'video_request.json'
        if not request.exists():
            return
        data=json.loads(request.read_text())
        request.unlink()
        if data['action']=='start':
            if self.active:
                raise RuntimeError('A recording is already active')
            if not re.fullmatch(r'[A-Za-z0-9_-]+',data['id']):
                raise ValueError('Invalid recording id')
            self.directory=self.video_directory/data['id']
            self.directory.mkdir(parents=True,exist_ok=False)
            self.frames=[];self.dropped=0;self.pending=[]
            self.identifier=data['id'];self.active=True
            self.started_wall=time.time()
            self.status('recording')
        elif data['action']=='stop' and self.active:
            self.finish()

    def status(self,state):
        data=dict(id=self.identifier,state=state,frames=len(self.frames),dropped_frames=self.dropped,
                  directory=str(self.directory))
        (self.sim_directory/'video_state.json').write_text(json.dumps(data))

    def capture(self, observer_rgb, wrist_rgb, sim_time):
        if not self.active:
            return
        self.pending=[f for f in self.pending if not f.done() or f.result()]
        if len(self.pending)>=4:
            self.dropped+=1
            return
        if observer_rgb.ndim!=3 or wrist_rgb.ndim!=3 or not observer_rgb.size or not wrist_rgb.size:
            return
        index=len(self.frames)
        # Copies isolate GPU annotator buffers from the background writer.
        observer=np.ascontiguousarray(observer_rgb[:,:,:3][:,:,::-1])
        wrist=np.ascontiguousarray(wrist_rgb[:,:,:3][:,:,::-1])
        frame=dict(index=index,wall_time=time.time(),sim_time=float(sim_time))
        self.frames.append(frame)
        def save():
            for name,image in [('observer',observer),('wrist',wrist)]:
                if not cv2.imwrite(str(self.directory/f'{index:06d}_{name}.jpg'),image,
                                   [cv2.IMWRITE_JPEG_QUALITY,95]):
                    raise IOError('Video frame write failed')
        self.pending.append(self.writer.submit(save))

    def finish(self):
        for future in self.pending:
            future.result()
        self.pending=[]
        (self.directory/'frames.json').write_text(json.dumps(self.frames))
        (self.directory/'capture.json').write_text(json.dumps(dict(
            id=self.identifier,frames=len(self.frames),dropped_frames=self.dropped,
            started_wall=self.started_wall,finished_wall=time.time(),
            source='Live Isaac Sim rendered frames; not reconstructed from trajectories'),indent=2))
        self.active=False
        self.status('finished')
