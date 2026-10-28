import json
from pathlib import Path
import sys
import numpy as np
root=Path(sys.argv[1]);summary={}
for mode in ['offline','live']:
    path=root/f'{mode}.json'
    if not path.exists():continue
    result=json.loads(path.read_text());all_rows=result['rows'];rows=[r for r in all_rows if r['success']]
    stats={}
    fields={key:[r['timings'][key] for r in rows] for key in rows[0]['timings']} if rows else {}
    fields['compute_s']=[r['compute_s'] for r in rows]
    fields['sam_total_s']=[r['timings']['sam_encoder_s']+r['timings']['sam_decoder_s'] for r in rows]
    if mode=='live':
        for key in ['wait_fresh_rgbd_s','frame_age_at_ready_s','request_to_pose_s']:
            fields[key]=[r[key] for r in rows]
    for key,values in fields.items():
        if values:stats[key]=dict(median_ms=float(np.median(values)*1000),p95_ms=float(np.percentile(values,95)*1000))
    summary[mode]=dict(mode=result['mode'],attempts=len(all_rows),completed_computations=len(rows),
        initialization_s=result['load_s'],stats=stats,
        sam_compute_fraction=None if not rows else sum(fields['sam_total_s'])/sum(fields['compute_s']),
        note='Completed computation is not proof of correct object localization or grasp success')
    if mode=='live' and rows:
        summary[mode]['sam_request_fraction']=sum(fields['sam_total_s'])/sum(fields['request_to_pose_s'])
(root/'summary.json').write_text(json.dumps(summary,indent=2))
if 'offline' in summary and 'live' in summary:
    a=summary['offline'];b=summary['live'];s=b['stats']
    lines=['识别、分割、RGB-D 几何与抓取参考位姿耗时（2026-10-28）','',
        '配置：同物理GPU3 RTX4090；YOLOv8s-worldv2、imgsz1280、conf=.02、四类词汇缓存；SAM ViT-B bbox提示。',
        '纯计算计时在CUDA阶段同步，包含预处理与输出计算，不含加载、磁盘I/O、IK、规划和机械臂移动。',
        f"离线：23请求×3次=69次，30次进入全部计算阶段（10个检测到的目标请求×3）；完整计算中位 {a['stats']['compute_s']['median_ms']:.2f}ms。",
        '未检出请求提前返回，未混入完整计算的延迟分布；“完成计算”不表示目标或位姿一定正确。',
        '离线覆盖之前腕部/外部相机已录制的同步RGB-D。',
        '', '在线：固定外部相机，观测静止绿块；只读传感器，不创建机械臂控制器。',
        '12次请求均得到计算结果；这不是12次定位精度验证或抓取成功测试。',
        '每次要求RGB、深度、标定时间戳完全匹配，且采集时间晚于本次请求；不是读取旧缓存。',
        '', '在线阶段耗时（中位 / P95，毫秒）：']
    for key in ['wait_fresh_rgbd_s','yolo_s','selection_s','sam_encoder_s','sam_decoder_s',
                'mask_geometry_s','grasp_transform_s','compute_s','request_to_pose_s']:
        lines.append(f"{key}: {s[key]['median_ms']:.3f} / {s[key]['p95_ms']:.3f}")
    lines += ['',f"SAM占在线纯计算时间 {b['sam_compute_fraction']*100:.1f}%，占请求到结果总时间 {b['sam_request_fraction']*100:.1f}%。",
        '各阶段中位数/P95不能直接相加成为总流程分位数；总流程按每次请求直接计时。',
        '等待项包含请求调度、下一次渲染/相机采集、发布传输和同步。在线另记录帧到达时年龄。',
        '模型初始化本次约3.6秒，仿真冷启动和首次预热也不计入暖启动数据；生产调用需让模型常驻。',
        '', '位姿范围：',
        '当前函数从掩码内有效深度点反投影，做异常值过滤、相机到世界坐标变换和稳健表面位置估计。',
        '使用当前方块策略的固定向下四元数，再加185mm TCP偏移并转换到规划基座坐标。',
        '这不是任意物体完整6D姿态估计，不是物体中心/接触点精度验证，也未验证逆解与路径可达。',
        '在线测试用斜视外部相机，可见表面点与物体中心有系统差异；不能直接把计时通过当成抓取位置正确。',
        '本次离线/在线探针使用单目标bbox提示SAM；生产 segment_all 多候选解码、关联、重试和日志可能增加开销。',
        '', '结论：当前已测纯计算中SAM图像编码最大；在线还需计入等新帧，等待在个别样本中可与编码相当。',
        '可优先评估SAM图像编码优化与观测调度。同一帧多个目标可复用图像编码，运动后新视角需要新编码。',
        '证据：offline.json / live.json / summary.json；脚本 tests/benchmark_pose_pipeline.py。']
    (root/'pose_timing_report.txt').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps(summary,indent=2))
