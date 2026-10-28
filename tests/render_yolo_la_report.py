import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(sys.argv[1]);rows=json.loads((root/'summary.json').read_text())
yolo640=next(r for r in rows if r['model']=='YOLO-World' and r['imgsz']==640 and r['conf']==.02)
yolo=next(r for r in rows if r['model']=='YOLO-World' and r['imgsz']==1280 and r['conf']==.02)
student=next(r for r in rows if r['model']=='student06_bboxonly')
teacher=next(r for r in rows if r['model']=='teacher_native_bboxonly')
sam=next(r for r in rows if 'SAM ViT-B' in r['model'])
selected=[yolo640,yolo,student,teacher]
labels=['YOLO-World\n640','YOLO-World\n1280','LA student\n0.6B','LA original\n3B hybrid']
colors=['#4c95af','#237696','#a9794b','#9469aa']
fig,axes=plt.subplots(1,2,figsize=(10.5,4.1))
times=[r['median_s']*1000 for r in selected]
axes[0].bar(range(4),times,color=colors)
axes[0].set_yscale('log');axes[0].set_ylabel('Median warm latency (ms, log scale)')
for i,t in enumerate(times):axes[0].text(i,t*1.12,f'{t:.1f}',ha='center',fontsize=9)
axes[0].set_ylim(3,3000)
accuracy=[r['localized'] for r in selected]
axes[1].bar(range(4),accuracy,color=colors)
axes[1].set_ylim(0,19);axes[1].set_ylabel('BBox IoU >= 0.5, out of 17 positive requests')
for i,value in enumerate(accuracy):axes[1].text(i,value+.35,f'{value}/17',ha='center',fontsize=9)
for ax in axes:
    ax.set_xticks(range(4),labels);ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
fig.suptitle('Matched color-category requests on 6 recorded frames; RTX 4090; bbox only',fontsize=11)
fig.text(.5,.005,'YOLO: one pass / four cached classes, conf 0.02 | LA: one RGB + single-target text query',ha='center',fontsize=8)
fig.tight_layout(rect=[0,.03,1,.96]);fig.savefig(root/'yolo_la_comparison.png',dpi=170)

text=['YOLO-World 与 LocateAnything 同帧感知对照（2026-10-28）','',
      '范围：同一组 6 张机械臂录制画面，17 条英文颜色类别正例 + 6 条不存在的黄色方块请求。',
      '不覆盖自由形式空间推理，不是完整自然语言抓放成功率对比。两者均未接管本次机械臂运动。',
      '模型输入全幅 RGB，不提供 HSV 提示；HSV 参照只评分。输入图像按原清单 SHA256 核验。',
      'YOLO 权重 yolov8s-worldv2.pt，缓存 red/green/blue/yellow cube 四类文本嵌入；',
      'LA 使用各自原生图文提示。两者输出均为 bbox，学生掩码头关闭。IoU>=0.5 记为定位达标。','',
      '热推理结果（模型和文本初始化、评分、可视化、存盘均不计）：']
for r in selected:
    label=r['model']+(' imgsz='+str(r['imgsz'])+' conf='+str(r['conf']) if r['model']=='YOLO-World' else '')
    text.append(f"{label}: 中位 {r['median_s']*1000:.2f}ms，P95 {r['p95_s']*1000:.2f}ms；框达标 {r['localized']}/17；无黄色时无框 {r['correct_absence']}/6。")
text+=['',f"YOLO 1280 + SAM ViT-B：检测到的目标上约 {sam['detected_median_s']*1000:.1f}ms/目标。",
       '这是分别测量的 YOLO 与完整 SAM 图像编码/掩码解码中位耗时之和，不是整段 ROS 抓放实测。',
       f"仅 10/17 个正例进入 SAM；其平均掩码 IoU={sam['mean_mask_iou_detected']:.3f}，漏检按0计入全部17例后为 {sam['mean_mask_iou_all']:.3f}。",
       'SAM 仅使用 YOLO 框提示，没有颜色或真值点提示。不能把只统计已检测目标的质量当作整体成功率。','',
       '阈值敏感性：YOLO 1280 在 conf=0.25 时仅 3/17 框达标；conf=0.02 为 9/17。',
       '因此 YOLO 也不是零样本即稳定，当前小目标和颜色检测仍需进一步验证与适配。','',
       '解释：本组 YOLO bbox 比学生 LA 约快 31 倍，比原始 LA 约快 116 倍。',
       '这些是当前配置的应用测量，不能作为两个模型家族在所有数据集上的架构速度/精度结论。',
       'YOLO 单次前向同时检测四类；LA 每次定位一段描述。YOLO=30 次计时/尺寸，LA=23 次单请求计时。',
       '硬件均为物理 GPU3 RTX4090；YOLO/SAM 使用已有 Docker torch2.4.1+cu121，LA 使用 conda la torch2.5.1+cu124。',
       '运行环境不同，属于现有可部署路径对比，不是同软件栈的严格算子微基准。',
       'LA 的词汇能力、关系推理和 YOLO 的检测候选能力不能由这 23 条类别请求互相替代。','',
       '工程建议：如果目标是先做可靠语言抓放，优先语言解析 -> 结构化目标 -> 检测候选 -> 深度几何关系 -> 分割 -> 现有反馈状态机。',
       '语言解析低频执行；类别嵌入缓存；模型常驻；局部跟踪与运动反馈独立。LA 可保留为复杂指代对照。',
       '工作站单次 0.3s/1.1s 未必是低频抓取的主瓶颈，但不适合放进每帧高频视觉伺服。',
       '当前 LA 学生更大的成本是达到目标域可靠性所需的数据、训练和验证，而非目前的单次 warm latency。',
       '未对 Jetson/RK3588 测速，未对微调所需样本数或训练时间作实测保证。','',
       '数据：manifest.json、protocol.json、summary.json；原始输出位于 yolo_world/ 与 *_bboxonly/。',
       'yolo_world_path_failure/ 是最初路径映射错误的无效运行，未计入任何表格。已增加图像存在与校验和断言。']
(root/'comparison_report.txt').write_text('\n'.join(text),encoding='utf-8')
print('Wrote comparison_report.txt and yolo_la_comparison.png')
