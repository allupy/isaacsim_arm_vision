# 环境与运行约定

## ws 现有环境

- 宿主机 Linux + Docker + NVIDIA Container Toolkit。
- 仿真镜像：`isaac-sim-omnilrs-0513:latest`（工作站自建镜像，未上传镜像仓库）。
- 控制镜像：`moveit_aluupy:latest`（ROS 2 Humble、Python 3.10、MoveItPy、视觉依赖）。
- ROS 工作空间默认 `/home/aluupy/omnilrs_ros2_demo/humble_ws`，挂载到 `/ros2_ws:ro`。
- 需要 `moveit_ws/install/setup.bash` 与 moveit_py 安装目录，MoveIt 配置源码快照见 `ros2/ur5_moveit_config/`。
- YOLO 权重默认 `/ros2_ws/isaacsim_arm_vision/yolov8s-worldv2.pt`，SAM 默认 `/ros2_ws/vision/sam_vit_b_01ec64.pth`。

启动前确认这两个权重文件存在。可通过 `ROBOT_YOLO_WEIGHTS`、`ROBOT_SAM_WEIGHTS` 指定容器内可读路径；权重未提交。SAM 反馈路径使用 vit_b，与原交互入口 YAML 的 vit_h 配置不同。

宿主机只做规则规划时使用 Python 标准库；资产校验使用标准库。单元检查需要 numpy、opencv、scipy；实时审计还需要 numpy。完整视觉与运动依赖由上述镜像提供。

## 启动参数

| 环境变量 | 默认值 / 用途 |
|---|---|
| `ARM_RUN_ROOT` | 当前仓库 `.runtime/$USER`，每用户独立可写结果 |
| `ROS_WS` | 外部已编译 ROS 工作空间 |
| `ROS_DOMAIN_ID` | 78；多人运行必须显式协调为不同值 |
| `SIM_NAME` / `CONTROL_NAME` | `arm-$USER-sim` / `arm-$USER-control` |
| `SIM_GPU` / `CONTROL_GPU` | 2 / 3；按空闲 GPU 设置 |
| `SIM_IMAGE` / `CONTROL_IMAGE` | 上述工作站镜像标签 |
| `SCENE_DIR` | 仓库 `scenes/workbench`，包含 workbench.usd |
| `CACHE_ROOT` | 运行目录下 cache，避免共用个人缓存 |

统一入口将当前克隆目录挂载到 `/experiment/src:ro`，结果目录挂载到 `/experiment`；YCB 资产单独只读挂载，生成的目的地标记写入结果目录。共享 ROS 工作空间只读。即使各自使用独立源码，也需要避免共用 ROS Domain、容器名和输出目录。

`arm.sh up` 会创建或启动指定名称的容器；已经存在的容器保留创建时的挂载、GPU、镜像与环境，改变这些参数后应使用全新的容器名。修改 runtime JSON 后停止并重新启动以重新加载。

仿真冷启动可能较慢；不要只凭 docker ps 判断相机就绪。查看 `sim/sim.log` 和 `sim/latest.json`。`agent --reset-scene` 等待仿真重置确认，有超时保护。

## 新机器

当前仓库包括源码、初始 USD、物体资产、配置和小型实录，不包括整个 ROS/MoveIt 编译环境、镜像、权重。新机器需要自行准备兼容的 Isaac Sim 5.0/ROS 2 Humble 环境和上述依赖；本次未验证从空白系统完整安装。原项目交互式入口的用法保存在 `docs/legacy/README-original.md`。

## 历史脚本

`sim_scene/start_feedback_*.sh`、`run_performance_trials.sh` 及部分 tests 是原实验现场脚本，保留 `/home/aluupy/arm_feedback_experiments/20261027` 和固定容器名，以便追溯旧证据。新同学请使用 `scripts/arm.sh`，或先调整历史脚本路径再运行。
