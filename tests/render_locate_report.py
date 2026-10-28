"""Render a compact, auditable report from finished offline inference results."""
import json
from pathlib import Path
import sys
import cv2
import numpy as np
root=Path(sys.argv[1])
summary=json.loads((root/'summary.json').read_text(encoding='utf-8'))
models=[('student06','Student 0.6B / final'),('student17','Student 1.7B / step 30000'),
        ('teacher_native','Original teacher / native hybrid')]
bbox_followup='--bbox512' in sys.argv[2:]
if bbox_followup:
    models=[('student06','Student 0.6B / coordinate tokens'),
            ('student06_head','BBox head h128 / 8K steps'),
            ('student06_head512','BBox head h512 / 40K steps')]
cases=[('long_home_000','Find a red cube'),('long_home_001','Find a green cube'),
       ('long_home_007','Find a yellow cube (absent)')]
rows=[]
for model,label in models:
    predictions=json.loads((root/model/'results.json').read_text(encoding='utf-8'))['cases']
    lookup={r['id']:r for r in predictions}
    cells=[]
    for identifier,prompt in cases:
        image=cv2.imread(str(root/model/f'{identifier}.jpg'))
        canvas=np.full((340,480,3),245,np.uint8)
        canvas[55:325]=cv2.resize(image,(480,270))
        row=lookup[identifier]
        cv2.putText(canvas,label,(8,17),cv2.FONT_HERSHEY_SIMPLEX,.48,(30,30,30),1,cv2.LINE_AA)
        cv2.putText(canvas,prompt,(8,36),cv2.FONT_HERSHEY_SIMPLEX,.48,(30,30,30),1,cv2.LINE_AA)
        score='Expected: no box' if row['target'] is None else f'Box IoU: {row["target_iou"]:.3f}'
        cv2.putText(canvas,score,(8,52),cv2.FONT_HERSHEY_SIMPLEX,.42,(30,30,30),1,cv2.LINE_AA)
        cells.append(canvas)
    rows.append(np.hstack(cells))
figure=np.vstack(rows)
footer=np.full((38,figure.shape[1],3),245,np.uint8)
cv2.putText(footer,'GREEN: diagnostic reference | RED: model box | BLUE: student mask | RGB + text only entered the model',
            (12,25),cv2.FONT_HERSHEY_SIMPLEX,.57,(35,35,35),1,cv2.LINE_AA)
cv2.imwrite(str(root/('bbox512_examples.png' if bbox_followup else 'grounding_examples.png')),np.vstack([figure,footer]))

lines=['机械臂自然语言抓取：蒸馏定位模型可行性验证（2026-10-28）',
       '', '结论：推理接口可接、暖启动速度可用；当前检查点零样本定位不达标，未接管机械臂。',
       '', '测试：6 张未修改的录制 RGB 帧，53 个图文请求，其中 41 个有目标、12 个无匹配目标。',
       '包含原夹爪与短指场景、腕部初始/精定位画面、外部提起观察、高分辨率画面。',
       '查询覆盖三色属性、图像坐标左右关系、中文属性/指令、黄色不存在、矛盾关系。',
       '这些请求共享 6 帧，不是 53 个独立场景；不能作为总体泛化准确率。',
       '参照框/掩码由已知彩色方块的 HSV 连通区域生成并人工抽查，只用于评分。',
       '模型输入只有全幅 RGB 和文本；未使用颜色框提示、真值框提示、颜色纠错或定向裁剪。',
       '定位达标定义：预测框与目标参照框 IoU>=0.5。',
       '单个有效输出框的中心落在目标彩色可见区域另作诊断，不等于可执行抓取点。',
       '学生自带掩码质量对可见 HSV 参照计算，不是独立人工像素标注。',
       '', '场景结果：']
for r in summary:
    if r['dataset']!='scene':continue
    lines.append(f"{r['model']}: 定位 {r['localized']}/{r['positive_n']}，中心命中 {r['center_hits']}/{r['positive_n']}，"
                 f"平均框IoU={r['mean_box_iou']:.4f}，中位耗时={r['latency_median_s']:.3f}s，P95={r['latency_p95_s']:.3f}s，"
                 f"无目标时无框={r['negative_no_box']}/{r['negative_n']}。")
    if r['mean_mask_iou'] is not None:lines.append(f"  平均预测掩码IoU={r['mean_mask_iou']:.4f}。")
lines+=['','原验证集 8 样本正控（RefCOCOg，固定 seed=42）：']
for r in summary:
    if r['dataset']=='controls':lines.append(f"{r['model']}: 框定位 {r['localized']}/8，平均框IoU={r['mean_box_iou']:.4f}。")
lines+=['','排查与边界：',
        '1. 与原项目 DistillDataset/eval_student 路径对比：prompt token、像素张量、图像网格完全相同（input_parity.json）。',
        '2. 检查点词嵌入形状：0.6B hidden=1024，1.7B hidden=2048，确认未误用同一模型（weight_audit.json）。',
        '3. 1.7B 检查点为 opt_step=30000，未完成原计划完整训练和校准，不能代表最终 1.7B 上限。',
        '   本轮未运行 V2 深度点指代分支；其已有基准结果未混入本次统计。',
        '4. 独立框头使用真实生成的 <box> 锚点；没有使用 teacher-forced 真值回复。',
        '5. 原始教师使用 native hybrid，净化教师使用 slow AR；权重与解码同时不同，不能用这两行单独归因某个因素。',
        '6. 教师也有较多失败，因此不能把差距全部归因于蒸馏容量。当前证据符合场景迁移不足，尚未隔离小目标、仿真风格与语言关系各自贡献。',
        '7. 所有模型在这组无目标请求中仍输出框。无框与明确拒绝分别计数；不能假设模型已有无目标拒绝能力。',
        '8. 暖启动延迟包含预处理、同步 GPU 推理及学生掩码头，排除模型加载、可视化和存盘；不是同一输出任务的纯网络架构速度对照。',
        '   框回归头探针仍先完整生成再重算锚点前向，未实现遇到 <box> 即停止的速度优化，不能据此否定原生 bbox 头的延迟收益。',
        '9. 未运行闭环抓取：第一阶段定位没有达到可执行筛选要求；没有把教师或颜色分割成功冒充学生抓取成功。',
        '', '下一步：先建立当前台面多布局、多尺寸、多实例的图文定位训练/独立留出集；加入无目标和关系冲突样本。',
        '分别验证框定位、目标关联与掩码，再恢复闭环抓放。不要仅凭框中心偶尔命中或输出格式合规就跳到运动执行。',
        '', '复现环境：conda la，PYTHONNOUSERSITE=1（屏蔽用户目录中不兼容的 huggingface-hub），',
        'CUDA_VISIBLE_DEVICES=3，MOONVIT_ATTN_IMPL=sdpa，HF_HUB_OFFLINE=1，TRANSFORMERS_OFFLINE=1。',
        '未升级共享依赖，未修改或训练任何模型权重。所有模型进程已结束。',
        '', '证据：summary.json / 各模型 results.json、cases.jsonl、可视化与掩码；controls/ 为正控；manifest.json 为固定测试输入。']
(root/'feasibility_report.txt').write_text('\n'.join(lines),encoding='utf-8')
print('Wrote visual comparison and feasibility_report.txt')
