#!/usr/bin/env python3
"""Interactive entry to the same feedback_runner.py closed-loop pipeline."""
import argparse
from contextlib import contextmanager
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import uuid
from modules.demo_command import parse_command

ROOT = Path(__file__).resolve().parent
HELP = '''
d / grasp     完整闭环抓放
o / s         只观察：保存同步图像、检测和定位结果，不创建运动控制器
t apple       选目标；也支持 t 红色方块
p fruit_zone  放置区：fruit_zone / cup_zone / plate / cup_stand
p x y z       自定义放置表面世界坐标（米）
c apple cup   自定义检测词；c auto 恢复按目标选择
plan          显示下一次执行参数，不运行
status        配置、相机状态与最近结果目录
last          最近结果与独立评分
v             相机快照（有桌面时显示，否则打印路径）
h / help      帮助
q / quit      退出
运行中 Ctrl+C 请求闭环保持并结束任务，然后返回菜单。
'''

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime-root',type=Path,default=Path('/experiment'))
    p.add_argument('--output-root',type=Path)
    p.add_argument('--config',default=str(ROOT/'config/default_config.yaml'))
    p.add_argument('--target','--instruction',dest='target')
    p.add_argument('--destination',choices=['fruit_zone','cup_zone','plate','cup_stand'])
    p.add_argument('--place-surface',type=float,nargs=3)
    p.add_argument('--classes',nargs='+')
    p.add_argument('--confidence','--det-confidence',type=float,default=.02)
    p.add_argument('--expected-gripper',choices=['short_parallel','robotiq_2f140'])
    p.add_argument('--tcp-offset',type=float,default=.185)
    p.add_argument('--home',type=float,nargs=3)
    p.add_argument('--startup-timeout',type=float,default=180.)
    p.add_argument('--sensor-timeout',type=float,default=25.)
    p.add_argument('--once',action='store_true')
    p.add_argument('--observe-only',action='store_true',help='整个会话禁用抓放，d 也只观察')
    p.add_argument('--dry-run',action='store_true')
    return p

class Console:
    def __init__(self,args):
        self.args=args
        if not 0<args.confidence<=1:raise ValueError('confidence must be in (0, 1]')
        for values in [args.place_surface,args.home,[args.tcp_offset,args.startup_timeout,args.sensor_timeout]]:
            if values is not None and not all(math.isfinite(v) for v in values):raise ValueError('参数必须为有限数字')
        if min(args.tcp_offset,args.startup_timeout,args.sensor_timeout)<=0:raise ValueError('偏移和超时必须为正数')
        path=args.runtime_root/'sim/runtime_config.json'
        runtime=json.loads(path.read_text()) if path.exists() else {}
        self.gripper=args.expected_gripper or runtime.get('gripper_profile','robotiq_2f140')
        if self.gripper not in ['short_parallel','robotiq_2f140']:raise ValueError('未知夹爪配置')
        self.target=parse_command(args.target or ('apple' if runtime.get('demo_scene')=='everyday' else 'red cube'))['target']
        self.classes=args.classes
        self.destination=args.destination
        self.place=args.place_surface
        if self.place and self.destination:raise ValueError('放置坐标与命名区域只能指定一个')
        self.output_root=args.output_root or args.runtime_root/'runs'
        self.last=None
        self.blocked=False

    def command(self,output,observe=False):
        destination=self.destination or ('fruit_zone' if self.target=='apple' else 'cup_zone' if self.target=='cup' else None)
        place=self.place or ([.35,-.30,.052] if self.target=='apple' else [.49,-.30,.058] if self.target=='cup' else [.4,-.3,.06])
        if not self.place and destination in ['fruit_zone','cup_zone']:
            place=([.35,-.30] if destination=='fruit_zone' else [.49,-.30])+[place[2]]
        if destination in ['plate','cup_stand'] and self.target not in ['apple','cup']:raise ValueError('承托物仅校准了 apple/cup')
        cmd=[sys.executable,'-u',str(ROOT/'feedback_runner.py'),'--output',str(output),'--config',self.args.config,
             '--perception','yolo_sam','--instruction','Pick the '+self.target,'--det-confidence',str(self.args.confidence),
             '--tcp-offset',str(self.args.tcp_offset),'--safe-z','.5','--expected-gripper',self.gripper,'--locked-sam-tracking',
             '--home',*[str(v) for v in (self.args.home or ([.45,-.12,.5] if self.gripper=='short_parallel' else [.30,.25,.5]))],
             '--place-surface',*[str(v) for v in place],'--startup-timeout',str(self.args.startup_timeout),
             '--sensor-timeout',str(self.args.sensor_timeout)]
        if self.classes:cmd+=['--classes',*self.classes]
        if destination in ['plate','cup_stand']:cmd+=['--destination-tag',str(10 if destination=='plate' else 20)]
        if observe or self.args.observe_only:cmd+=['--observe-only']
        return cmd

    @contextmanager
    def lock(self):
        path=self.args.runtime_root/'sim/agent_execution.lock'
        path.parent.mkdir(parents=True,exist_ok=True)
        try:path.mkdir()
        except FileExistsError:raise RuntimeError('场景已被交互任务或 agent 占用，请检查运行状态') from None
        try:yield
        finally:path.rmdir()

    def run(self,observe=False):
        if self.blocked:raise RuntimeError('上次退出后运动状态未知；检查控制器后重新启动会话')
        identifier='interactive_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]
        output=self.output_root/identifier
        cmd=self.command(output,observe)
        print('执行：'+shlex.join(cmd),flush=True)
        self.output_root.mkdir(parents=True,exist_ok=True)
        with self.lock(),(self.output_root/(identifier+'.console.log')).open('w',encoding='utf-8') as log:
            process=subprocess.Popen(cmd,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,
                                     encoding='utf-8',errors='replace',start_new_session=True)
            interrupted=False
            while True:
                try:
                    line=process.stdout.readline()
                    if not line:
                        code=process.wait();break
                    print(line,end='',flush=True);log.write(line);log.flush()
                except KeyboardInterrupt:
                    if not interrupted and process.poll() is None:
                        process.send_signal(signal.SIGINT);interrupted=True
                    print('正在等待闭环保持和清理',flush=True)
        self.last=output
        if output.exists():(output/'invocation.json').write_text(json.dumps(cmd,ensure_ascii=False,indent=2),encoding='utf-8')
        path=output/'result.json'
        result=json.loads(path.read_text()) if path.exists() else {}
        if not result or code not in (0,2,130) or result.get('state_known') is False or result.get('reason','').startswith('runtime_error'):
            self.blocked=True
        if interrupted and result.get('reason')!='interrupted':self.blocked=True
        print(f'返回码 {code}；结果目录：{output}',flush=True)
        return code

    def status(self):
        print(json.dumps(dict(target=self.target,classes=self.classes or 'auto',destination=self.destination or 'auto',
                             place_surface=self.place,gripper=self.gripper,observe_only=self.args.observe_only,
                             blocked=self.blocked,last=str(self.last) if self.last else None),ensure_ascii=False,indent=2))
        path=self.args.runtime_root/'sim/latest.json'
        print(f'相机状态：{path}；'+('存在，请结合时间戳判断新鲜度' if path.exists() else '未找到，请先启动反馈仿真桥接'))

    def dispatch(self,line):
        words=shlex.split(line)
        if not words:return True
        key,*values=words
        if key in ['q','quit']:return False
        if key in ['h','help']:print(HELP)
        elif key in ['d','grasp']:self.run()
        elif key in ['o','s']:self.run(observe=True)
        elif key=='t':
            self.target=parse_command(' '.join(values))['target'];self.classes=None
        elif key=='p':
            if len(values)==1 and values[0] in ['fruit_zone','cup_zone','plate','cup_stand']:
                self.destination,self.place=values[0],None
            elif len(values)==3:
                place=list(map(float,values))
                if not all(math.isfinite(v) for v in place):raise ValueError('坐标必须为有限数字')
                self.place,self.destination=place,None
            else:raise ValueError('使用 p fruit_zone 或 p x y z')
        elif key=='c':
            if not values:raise ValueError('使用 c auto 或 c apple cup')
            self.classes=None if values==['auto'] else values
        elif key=='plan':print(shlex.join(self.command(self.output_root/'PREVIEW_ONLY')))
        elif key=='status':self.status()
        elif key=='last':print((self.last/'result.json').read_text() if self.last and (self.last/'result.json').exists() else '没有结果')
        elif key=='v':
            paths=[self.args.runtime_root/'sim'/f'{name}.png' for name in ['observer','wrist']]
            print('\n'.join(map(str,paths)))
            if os.environ.get('DISPLAY') and os.environ.get('QT_QPA_PLATFORM')!='offscreen':
                import cv2
                for path in paths:
                    frame=cv2.imread(str(path))
                    if frame is not None:cv2.imshow(path.stem,frame)
                cv2.waitKey(0);cv2.destroyAllWindows()
        else:raise ValueError('未知命令；输入 h 查看帮助')
        return True

def main(argv=None):
    args=parser().parse_args(argv)
    try:
        console=Console(args)
        if args.dry_run:
            print(shlex.join(console.command(console.output_root/'PREVIEW_ONLY')));return 0
        if args.once:return console.run()
        print('视觉机械臂交互台 · 与 feedback_runner.py 共用闭环执行流程')
        print(HELP);console.status()
        while True:
            try:
                if not console.dispatch(input('arm> ')):break
            except (EOFError,KeyboardInterrupt):break
            except (ValueError,RuntimeError,OSError) as exc:print('错误：'+str(exc))
        return 0
    except (ValueError,RuntimeError,OSError) as exc:
        print('错误：'+str(exc),file=sys.stderr);return 2

if __name__=='__main__':raise SystemExit(main())
