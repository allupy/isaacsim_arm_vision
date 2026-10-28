# 机械臂反馈策略：实现与实验记录

日期：2026-10-27（Asia/Shanghai）。本目录为独立实验候选版本，从工作站 `humble_ws/isaacsim_arm_vision` 当前源码复制，保留其已有修改。没有覆盖原机械臂项目或 lunar-car。

2026-10-28 最新结果：已完成执行提速与卡顿修复。同场景每组 3 次，最终版本耗时中位数 10.72 秒现实时间，独立落点误差 3.83–5.17 mm。见 [提速、卡顿诊断与重复实验](PERFORMANCE_20261028.md)。下文是 2026-10-27 的历史记录。

观察位按手眼标定让相机居中，见 [相机居中观察改版及实测](CAMERA_CENTERED_VIEW.md)。

## 2026-10-27 结论

反馈状态机已在后续诊断中跑通一次完整的 Isaac Sim 抓取、提升、搬运与放置：`feedback_color_05`。该轮使用颜色分割诊断路径，耗时约 70.9 秒，独立真值评价的落点 XY 误差为 50.3 mm，满足当前 60 mm 区域容差。完整诊断、初始化/时序/IK 分支/自遮挡修复和证据见 [后续诊断报告](DIAGNOSIS_20261027.md)。下面保留前期实验记录；不能把单轮成功或软件测试通过率当作稳定抓取成功率。

## 新流程

`HOME → ACQUIRE → APPROACH → REFINE → DESCEND → CLOSE → LIFT → VERIFY_LIFT → TRANSFER → PLACE → RELEASE → RETREAT → VERIFY_PLACE → SUCCESS`

- 初次获取目标后锁定类别、初始外观和三维位置。后续关联使用位置门限与外观，不因另一个目标检测分数变高而换目标；多个候选难以区分时拒绝继续。
- 实验传感器发布同一采样批次的 RGB、深度和相机外参；按时间戳精确配对。每次重新观测必须获取调用之后采集的新帧，拒绝旧缓存。外参来自仿真相机标定，物体真值不进入策略。
- 深度使用掩码内部有效点的稳健统计，避免仅依赖质心一个像素。
- 使用 MoveIt 的路径规划和笛卡尔路径服务；路径不完整或规划失败时停止，不降级为单点 IK 强行执行。按关节名称解释 IK 结果。
- 发布轨迹后等待实际关节反馈收敛；不以命令目标覆盖观测值。夹爪闭合仅表明执行了闭合，之后仍需独立视觉证据。
- 提起与放置均需至少两个不同时间戳的观测。看不到物体、落点错误或高度不足不能判成功。
- 最多两次抓取尝试。只有重新确认原目标仍在原支撑面上，才允许空夹重试。搬运中失败时停止并保持夹爪。
- 每个状态、观测、运动目标、失败原因都写入 JSONL；RGB、深度和标定一起保存，物体真值单独记录供离线评价。
- 修复配置类忽略 `controller`、`two_stage_grasp` 等 YAML 字段的问题，并显式使用 UTF-8。
- 实验轨迹转换器收到单点停止/夹爪命令后会取消旧回放，避免旧线程继续覆盖新命令。

## 实验 A：确定性故障注入

执行原 `arm_manager.py` 的真实 `execute_pick_place` 方法（AST 提取），将运动/夹爪接口替换为总是返回 True 的测试接口。比较新策略在对应合成观测下的结果。这里验证的是完成判据，不是动力学。

| 合成情形 | 旧流程返回成功 | 反馈流程返回成功 | 解释 |
|---|---:|---:|---|
| 正常提起并放置 | 是 | 是 | 需满足提起和放置验证 |
| 空夹 | 是 | 否 | 两次有限尝试后停止 |
| 物体遮挡、状态不可知 | 是 | 否 | 不把缺少观测解释为成功 |
| 放错位置 | 是 | 否 | 不满足目标区域验证 |

详细结果：`results/fault_injection/summary.json`、`summary.csv` 及逐场景状态日志。

另有 17 项单元测试，覆盖同步/旧帧、无效深度、目标混淆、外观漂移、有限重试、运动失败、搬运中保持夹爪，以及 YAML 配置回归。

## 实验 B：真实 Isaac Sim 传感器诊断

运行环境：独立 ROS_DOMAIN_ID=78；仿真 GPU 2，感知 GPU 3。使用现有 workbench USD；原文件只读挂载。

`sensor_color_02` 使用颜色分割诊断可见彩色方块，按同一时间戳的仿真真值离线评分：

| 相机 | 样本 | XY 平均误差 | XY 最大误差 |
|---|---|---:|---:|
| 腕部斜视 | 一个帧中的 2 个方块 | 26.0 mm | 27.6 mm |
| 顶视观察相机 | 一个帧中的 3 个方块 | 4.0 mm | 4.8 mm |

位置估计是可见表面的中值，XY 与方块中心的水平投影比较。斜视下可见侧面会造成几何偏差。本实验采用理想仿真标定、样本很少，只用于定位问题；**颜色分割结果不代表开放词汇检测表现**。

原始数据和逐目标评分：`results/isaac/runs/sensor_color_02/`。

## 实验 C：真实控制与开放词汇感知

- `feedback_color_01`：MoveIt 生成归位轨迹，但实际关节在约 46 秒的归位步骤内未收敛，最终记录的最大关节误差约 10.397 rad。策略在 HOME 阶段停止，未进入抓取；因此此轮不能统计为一次抓取尝试。该误差是当时实现采用的关节角绝对差，不等同于末端位置误差。
- `sensor_yolo_01`：YOLO-World v2，类别 `cube, block`，置信阈值 0.02，输入尺寸 1280，SAM ViT-B。在该轮腕部和观察相机帧中均没有产出有效三维目标。不能据此声称模型总体失效，也没有开放词汇抓取成功证据。
- 首次传感器试验 `sensor_color_01` 在仿真尚未就绪时超时。保留失败结果，未当作目标检测负样本。

源码检查还发现：场景的部分非零初始关节状态配着零驱动目标，可能导致播放后先回零；相机安装姿态与原手眼参数不符。实验脚本包含仅内存生效的初始化与相机安装修正候选，默认关闭（`REPAIR_PARKED_POSE`、`REPAIR_CAMERA_MOUNT`）。修正后的重启遇到 Isaac Sim 原生崩溃，尚未完成验证；它们不改变上述已采集数据。驱动、坐标系、关节限位以及低仿真速度与墙钟轨迹回放之间的关系仍需进一步验证。

## 复现

本地软件测试（依赖 numpy、opencv-python、PyYAML）：

```powershell
python -m unittest discover -s D:/school/arm-feedback/tests -p test_feedback.py -v
python D:/school/arm-feedback/tests/run_fault_experiments.py
python D:/school/arm-feedback/tests/score_observations.py D:/school/arm-feedback/results/isaac/runs/sensor_color_02
```

工作站实验目录：`/home/aluupy/arm_feedback_experiments/20261027/`。`baseline.tgz` 是原始源码快照，`preexisting.diff` 是开始时已有修改。

启动脚本只针对这个实验目录和独立容器；再次启动前检查已有同名容器状态，不要同时启动多个轨迹发布者：

```bash
bash src/sim_scene/start_feedback_sim.sh
bash src/sim_scene/start_feedback_control.sh
docker exec arm-feedback-control-20261027 bash /experiment/src/sim_scene/run_feedback.sh \
  --output /experiment/runs/new_unique_run --perception color \
  --home 0.4 0.05 0.5 --place-surface 0.4 -0.3 0.06
```

`--perception yolo_sam` 是开放词汇检测路径，`color` 是彩色方块诊断路径。运行目录必须是新目录，避免覆盖历史证据。`--observe-only` 不发送动作。`--trajectory-time-scale` 可用于独立诊断低仿真速度，但不替代仿真时钟驱动的轨迹执行。

停止本次两个实验容器：`bash src/sim_scene/stop_feedback.sh`。不要停止工作站其他容器。

## 尚未完成的物理验证

下一步应先解决归位执行不收敛，再确认相机安装、抓取 TCP 偏移和夹爪实际接触。之后固定场景版本、初始姿态、物体布局与检测配置，对旧流程和新状态机做成对试验，分别统计目标获取率、提起率、放置率、重试次数、错误成功报告率和耗时。当前证据不足以报告物理抓取成功率提升。

现有 MoveIt 模型不含完整夹爪；实验适配器也尚未把全部场景障碍和被抓物体作为附着碰撞体同步给规划场景。故这一版本属于仿真实验候选，不是已验证的实物部署版本。
