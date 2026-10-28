# 2026-10-28 源码统一

原仓库：`/home/aluupy/omnilrs_ros2_demo/humble_ws/isaacsim_arm_vision`。
实验来源：`/home/aluupy/arm_feedback_experiments/20261027/src`。

原仓库 `c36c699` 及以前的提交完整保留，随后提交 `0794bc2` 保存开始整理时已有的检测器和配置修改。本次将实验源码并入仓库根目录，原入口 `main.py` 与反馈入口 `feedback_runner.py` 同仓管理。旧 README 见 `docs/legacy/README-original.md`。

新增 `scripts/arm.sh` 是可配置的统一启动入口：宿主机路径、容器名、GPU、ROS Domain ID 可通过环境变量设置；容器内保持 `/experiment/src`、`/experiment/sim`、`/ws` 约定，以兼容现有代码。代码从当前克隆目录只读挂载，运行结果写入独立 `.runtime/<user>/`。

`sim_scene/` 和 `tests/` 内的历史诊断、基准脚本保留了当时的工作站路径、容器名和日期，主要用于追溯，不能视为可移植的一键入口。已有实验目录的源码、视频、结果不删除；原始基线和大结果归档仍保留在工作站。

历史报告记录的是当时的实验，不代表合并版本已经重新跑过全部仿真。README 中列出的实录和评分均附原始运行标识；新提交的检查范围见 `docs/VALIDATION.md`。
