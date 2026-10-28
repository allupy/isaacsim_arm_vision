"""Append the h512 follow-up correction to the project's experiment log."""
from pathlib import Path
import sys
import json
root=Path(sys.argv[1]);summary=json.loads((root/'summary.json').read_text())
scene=next(r for r in summary if r['dataset']=='scene' and r['model']=='student06_head512')
control=next(r for r in summary if r['dataset']=='controls' and r['model']=='student06_head512')
path=Path('/mnt/intel_ssd/locateanything/docs/EXPERIMENT_LOG.md')
marker='[10-28] 补测 h512/40K bbox 回归头'
if marker in path.read_text():raise RuntimeError('Follow-up already logged')
text=f'''
- **{marker}**：用户指出 bbox 专用版本；上一轮只测 h128/8K，
  不能概括全部回归头。补测 `work_dirs/boxhead_calib56K_h512/ckpt_head.pt`
  （hidden=512，40K步，base=`calib56K_plus_head/student_init.safetensors`）。
  沿用同一固定清单 53 请求：有目标定位 IoU>=0.5 为 {scene['localized']}/41，
  mean IoU={scene['mean_box_iou']:.5f}，中心命中 {scene['center_hits']}/41；
  12无匹配请求均仍输出框。原验证集8样本正控 {control['localized']}/8，mean IoU={control['mean_box_iou']:.5f}。
  使用真实生成的 <box> 锚点，非 teacher-forced GT 回复。当前台面未改善到可执行程度；
  不否定历史 n=500 原域 h512 优于 token 的结果。头探针包含完整生成+额外锚点前向，
  不能用本次耗时判定省坐标token后的最佳部署速度。未训练权重、未驱动机械臂。
  证据仍存外部机械臂目录的 `diagnostics/locate_feasibility_20261028/student06_head512`。
'''
with path.open('a') as stream:stream.write(text)
print('Recorded h512 follow-up and clarified prior coverage')
