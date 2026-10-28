import json
from pathlib import Path
import sys
root=Path(sys.argv[1]);rows=json.loads((root/'summary.json').read_text())
log=Path('/mnt/intel_ssd/locateanything/docs/EXPERIMENT_LOG.md')
marker='[10-28] YOLO-World 与 LA 同帧类别定位/耗时对照'
if marker in log.read_text():raise RuntimeError('Already recorded')
lines=['',f'- **{marker}（非完整抓放对比）**：',
    '  同6帧中选英文颜色类别17正例+黄色不存在6负例；RGB校验和一致；参照只评分。',
    '  同物理GPU3 RTX4090，YOLO/SAM沿用Docker torch2.4.1，LA沿用conda torch2.5.1，',
    '  属于现有可部署路径比较，不是同软件栈严格算子基准。bbox-only暖启动、评分/存盘/加载不计。']
for row in rows:
    if 'median_s' not in row:continue
    detail=f" imgsz={row['imgsz']} conf={row['conf']}" if row['model']=='YOLO-World' else ''
    lines.append(f"  - {row['model']}{detail}：中位{row['median_s']*1000:.2f}ms，P95 {row['p95_s']*1000:.2f}ms，定位{row['localized']}/17，无黄色无框{row['correct_absence']}/6。")
lines+=['  YOLO四类嵌入提前缓存，一次前向覆盖四类；LA一次单目标图文请求。',
    '  额外SAM ViT-B采用预测框提示，每次完整图像编码；YOLO1280+SAM分阶段中位数之和约83.1ms/已检测目标。',
    '  仅10/17正例进入SAM，掩码meanIoU已检0.888，全17例漏检计0后0.522，不能只报条件指标。',
    '  YOLO在conf=.02仍仅9/17，不能当作已稳定的语言抓取方案；本轮不比较自由形式空间理解。',
    '  判断：LA低频选目标时间未必致命，高频视觉闭环开销明显；当前学生更主要成本在目标域适配。',
    '  未改权重、未训练、未运行模型驱动抓放。',
    '  证据：`/home/aluupy/arm_feedback_experiments/20261027/diagnostics/yolo_la_compare_20261028`。','']
with log.open('a') as stream:stream.write('\n'.join(lines))
print('Recorded matched YOLO/LA comparison')
