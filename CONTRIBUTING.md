# 同学协作

1. 每人独立 clone；在 `feat/<任务名>` 或 `fix/<问题名>` 分支开发。
2. 一个 Issue 对应一个可验收任务，写清输入、输出、场景、评价指标。
3. 提交合并请求，附检查结果、运行配置、Git 提交号和实验输出位置，由项目负责人检查后合入 master。
4. 共用 ws 时为每人设置独立 ARM_RUN_ROOT、ROS_DOMAIN_ID、SIM_NAME、CONTROL_NAME，协调 GPU 使用；不要并发控制同一组仿真。
5. 源码、场景、配置、脚本、小型证据进 Git；密钥用环境变量传入，权重、缓存和整批原始录制单独保存。

| 方向 | 主要代码 | 首个验收任务 |
|---|---|---|
| 视觉定位 | detection_manager / feedback_observation / camera_viewpoint | 固定距离、视角，报告坐标误差 |
| 运动与夹爪 | feedback_motion / trajectory_playback / short_parallel_gripper | 改善一种失败，提供前后对照 |
| 场景与评测 | sim_scene / tests / 独立评分 | 多摆放重复试验，输出成功率与失败类型 |
| 整体集成 | feedback_policy / agent_workflow / 配置 | 固定基线和接口、审查合并 |

每次实验记录提交号、场景配置、感知模式、随机种子（适用时）、重复次数、控制器返回结果和独立评分。颜色诊断与 YOLO+SAM 的结果分开统计。
