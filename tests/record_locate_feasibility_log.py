"""Append the required project experiment-log entry, preserving prior records."""
import json
from pathlib import Path
import sys
root=Path(sys.argv[1])
summary=json.loads((root/'summary.json').read_text())
log=Path('/mnt/intel_ssd/locateanything/docs/EXPERIMENT_LOG.md')
marker='[10-28] 机械臂录制场景零样本可行性验证'
previous=log.read_text()
if marker in previous:
    raise RuntimeError('This experiment-log entry already exists; refusing duplication')
lines=['',f'- **{marker}（只读权重，未训练）**：',
    '  在独立机械臂实验目录，对 6 张已录制 RGB 帧固定 53 个图文请求',
    '  （41 有目标 + 12 无匹配目标；多请求共享画面，不是 53 个独立场景）。',
    '  模型只输入全幅 RGB+文本；已知色块 HSV 参照仅评分，未提供真值框/掩码提示。',
    '  定位达标口径为 bbox IoU>=0.5：']
for row in summary:
    if row['dataset']=='scene':
        lines.append(f"  - {row['model']}: {row['localized']}/{row['positive_n']}，框平均IoU {row['mean_box_iou']:.4f}，暖启动中位耗时 {row['latency_median_s']:.3f}s。")
lines+=['  各模型 12 个无目标请求均仍输出框；原 RefCOCOg 8 样本正控：',
        '  0.6B=5/8、1.7B=5/8、0.6B框头=3/8、净化教师=6/8、原始教师hybrid=2/8。',
        '  输入与原 DistillDataset/eval_student 的 token、像素张量、网格逐项一致。',
        '  权重形状核对：0.6B hidden=1024；1.7B hidden=2048、opt_step=30000（中间检查点，非终评模型）。',
        '  原始教师用hybrid、净化教师用slow，不能将两行差异单独归因权重或解码。',
        '  判断：接口/速度可接，当前这些学生权重零样本台面定位不达标；未接管机械臂。',
        '  结果符合场景迁移不足，尚未隔离小目标/仿真风格/关系能力的各自贡献。V2本轮未运行。',
        '  环境须 PYTHONNOUSERSITE=1，屏蔽用户目录不兼容 huggingface-hub；未升级共享依赖。',
        '  完整证据：`/home/aluupy/arm_feedback_experiments/20261027/diagnostics/locate_feasibility_20261028/`。',
        '  复现入口：同根 `src/tests/locate_feasibility.py`、`run_locate_feasibility.sh`、`run_locate_additional.sh`。','']
with log.open('a') as stream:stream.write('\n'.join(lines))
print('Appended one application-validation entry; prior experiment log preserved')
