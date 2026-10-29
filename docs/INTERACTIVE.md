# 闭环交互演示与调试

`main.py` 现在只管理交互、参数、任务互斥与结果展示。每次观察或抓放创建独立子进程，直接执行仓库的 `feedback_runner.py`；不再执行旧两阶段抓取控制代码。启动目录可以任意，配置默认指向仓库文件。

## 启动

在 ws 宿主机设置自己的容器名、ROS Domain、GPU 后：

```bash
source .env.example
bash scripts/arm.sh up
bash scripts/arm.sh interactive
```

已在反馈控制容器内时：

```bash
bash /experiment/src/sim_scene/run_python.sh main.py
```

本入口要求配套的 `run_feedback_sim.py`：除了图像，还需同步标定、观测请求和轨迹执行协议。不能只启动旧场景的 `/rgb`、`/depth` 发布器。初始场景与夹爪来自 `/experiment/sim/runtime_config.json`；日常物体默认 apple/短夹爪，方块场景默认 red cube/原夹爪。夹爪不匹配时执行器拒绝运行。

## 演示操作

```text
status          查看当前配置与相机状态文件
t 苹果          选择目标（也接受 apple、杯子、红色方块等）
p fruit_zone    选择固定放置区
plan            检查实际执行参数
o               只观察，保存检测/定位证据
d               完整闭环抓取、提起、放置并验证
last            查看策略结果与独立评分
t 杯子
p cup_zone
d
q
```

`p x y z` 指定放置表面世界坐标，单位米；`p plate` / `p cup_stand` 使用 ArUco + 深度定位承托物，仅支持已校准的 apple/cup。改变目标后检测词恢复自动选择；改变目标不会清除用户手动设置的目的地，请用 `status` / `plan` 检查。

`c apple banana cup` 覆盖检测词，`c auto` 恢复自动选择。`v` 打开当前全景和腕部快照，按键关闭；无 DISPLAY 或 Qt offscreen 环境下打印 PNG 路径，可从宿主机查看。

`--observe-only` 对整个会话生效，即使输入 `d` 也不会初始化运动控制器。`--once` 执行一次后返回真实退出码；`--dry-run` 只输出调用命令。

```bash
python3 main.py --once --observe-only --target apple
python3 main.py --once --target cup --destination cup_zone
python3 main.py --dry-run --target "red cube"
```

## 调试与中断

每次操作使用新运行 ID，保存到 `/experiment/runs/interactive_*`，控制台完整日志在旁边的 `.console.log`。目录中包含调用参数、观测图像、事件、result.json 和闭环执行器生成的评分。策略成功和独立真值评分需分别查看。

操作串行执行，期间只接受 Ctrl+C 中断；执行器在 ROS 仍可用时尝试发送保持指令，保存中断结果并清理，再返回菜单。关节观测过期或子进程异常退出时，状态标记为未知，会话禁止继续发起任务，应先检查仿真与控制器。该保持命令不等同于硬件急停。

交互台与 agent_workflow 使用同一个 `sim/agent_execution.lock`。直接运行其他历史运动脚本不遵守此锁，需避免并发控制同一场景。

每条命令重新初始化视觉模型，因此相邻操作之间有加载延迟；这样的进程隔离可保留每轮参数和失败现场。旧 `main.py` 实现可以从 Git 提交 `a9bc00e` 查阅。

## 2026-10-29 验证范围

44 项本地检查通过（反馈策略 23、轨迹 5、agent 4、入口配置 5、交互命令/互斥 5、多用户容器 2），以及控制容器内 1 项中断回归（分别验证保持可用/不可用两种情况）。检查覆盖真实 feedback_runner 的中断清理、退出码与结果记录，不使用运动服务。

另起独立 ROS Domain 92、GPU 2/3 的仿真/控制实例尝试只观察接入，USD 在 `ctx.open_stage` 加载阶段未就绪，闭环返回 `startup_sensor_not_ready`；交互入口记录错误并禁止后续任务。使用已有缓存重试仍未完成场景加载。本轮未完成实时抓放复验，也未改动其他人的仿真实例。验证日志保留在 ws `/home/aluupy/project_sync_backups/20261029/`。
