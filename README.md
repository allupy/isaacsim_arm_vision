# isaacsim_arm_vision · 视觉机械臂抓取(Isaac Sim 版)

在 Isaac Sim 仿真中对 UR5 机械臂实现「开放词汇目标检测 → 分割 → 6D 定位 → 抓取放置」的完整流程。
实物相机版本见 `../vision/`(两项目共享同一套 modules,参数各有差异)。

## 系统流水线

```
Isaac Sim(动作图发布 /joint_states,订阅 /joint_command)
        │ ros2
        ▼
MoveIt(/compute_ik 服务,提供 IK 求解)
        ▲                            ┌────────────────────────────────┐
        │ service                    │  main.py (本程序)              │
/joint_states ◄──────────────────────│  YOLO-World 检测(开放词汇)     │
        + /rgb /depth(TF)            │  → SAM 分割(仅最优目标)        │
                                    │  → 掩码质心+深度 → 基座系坐标   │
/moveit_trajectory ─(转换器)→ /joint_command  ←  抓取/放置轨迹 ─────┘
```

## 运行

```bash
# 终端1: 启动 Isaac Sim(机械臂场景,含动作图)
# 终端2: 启动 MoveIt(同一 ROS 容器内)
ros2 launch ur5_moveit_config start_moveit_isaac.launch.py
# 终端3: 本程序(同一 ROS 容器内,cwd = isaacsim_arm_vision)
python3 main.py
```

快捷键:`d` 检测并抓取最优目标 / `s` 拍照保存 / `c` 改类别 / `r` 归位 / `l` 历史 / `q` 退出。

> 运行 MoveItPy 模式前需 source moveit_ws:
> `source /ros2_ws/moveit_ws/install/setup.bash` 且
> `export PYTHONPATH=/ros2_ws/moveit_ws/install/moveit_py/local/lib/python3.10/dist-packages:$PYTHONPATH`

## 两种控制方案(config: `controller`)

| | `controller: "moveit"`(推荐) | `controller: "ik"`(旧方案) |
|---|---|---|
| 自由空间运动 | **MoveItPy/OMPL 整条轨迹规划**(时间最优参数化,平滑无启停) | IK 服务逐点求解 + 手写圆弧规避 |
| 垂直下降/提升 | 直线插值 + 逐点IK(整条轨迹一次发布,保证垂直接近方向) | 同左 |
| 失败处理 | 规划失败自动回退 IK 单点模式;初始化失败自动降级 `ik` | — |

`modules/moveit_controller.py` 中 `build_ur5_moveit_config()` 封装了本版本 moveit_py(2.5.x)
初始化所需的三个配置修复——这也是当初 MoveItPy 方案跑不起来的真正原因:

1. `config_dict` 必须传**完整**的 MoveItConfigsBuilder 字典(传空字典等于没有机器人描述);
2. builder 生成的 `ompl` 段是双层嵌套(`ompl.ompl.planning_plugin`),需展平为 `ompl.planning_plugin`;
3. `planning_pipelines` 需转为命名空间形式 `{pipeline_names: [...]}`(即 moveit_cpp.yaml 的格式);
4. 本版本 `PlanRequestParameters` 从 MoveItPy 实例构造后逐字段赋值,`plan(parameters=...)` 传参。

## 轨迹执行链路(重要)

```
main.py / MoveItArmController ──/moveit_trajectory(JointTrajectory,整条)──►
trajectory_converter(ur5_moveit_config/launch/) ──60Hz 插值回放──► /joint_command ──► Isaac Sim
```

- 转换器已升级为**整条轨迹按时戳插值回放**(原版只转发第一个点,任何多点轨迹都会被截断);
- 回放用**墙钟线程**实现:launch 传入的 `use_sim_time:=true` 在无 /clock 时会冻结 ROS 时间,
  节点定时器永不触发(这是调试中实际踩到的坑);
- 单点轨迹仍立即发布,与旧命令格式完全兼容。

## 两阶段抓取(config: `two_stage_grasp: true`)

提高抓取准确率最简单有效的手段:

1. 第一次检测估计物体位置 → 运动到物体**正上方**;
2. 再次检测+位姿估计(距离近、视角正,误差显著更小);
3. 用第二阶段的估计执行抓取(任一阶段失败自动沿用上一阶段结果)。

## 无头自测脚本(不需要 Isaac Sim)

```bash
# 1. MoveItPy 初始化+规划测试(不需要任何外部节点)
python3 tests/test_moveitpy.py
# 2. 端到端测试(需 move_group+converter 在运行,用假 /joint_states 代替仿真)
ros2 launch ur5_moveit_config start_moveit_isaac.launch.py   # 另一终端
python3 tests/test_e2e.py
```

## 目录结构

```
├── main.py                 主节点:订阅/显示/位姿估计/命令循环
├── config/
│   ├── config.py           DetectorConfig(YAML 加载 + 权重路径解析)
│   └── default_config.yaml 全部可调参数(模型/内参/手眼外参/机械臂)
├── modules/
│   ├── detection_manager.py  YOLO-World + SAM(仅最优目标分割)
│   ├── arm_manager.py        机械臂控制基础类(IK/直线插值/夹爪/抓放流程)
│   ├── moveit_controller.py  MoveItPy 版控制器(OMPL 规划,继承 arm_manager)
│   ├── io_manager.py         结果保存与历史
│   ├── class_manager.py      开放词汇类别管理(伪自然语言交互)
│   └── keyboard_handler.py   非阻塞键盘输入
├── tests/                  无头自测脚本(test_moveitpy / test_e2e)
└── archive/                历史版本(废弃的 MoveItPy 初版、旧备份等)
```

## 配置要点(config/default_config.yaml)

| 参数 | 说明 |
|---|---|
| `model_path` / `sam_checkpoint` | 权重路径,按 当前目录→项目根→../vision 顺序查找 |
| `camera_matrix` / `dist_coeffs` | 相机内参(D455 1280×720) |
| `R_cam2gripper` / `t_cam2gripper` | 手眼标定结果(眼在手外,装于末端) |
| `arm.home_position` 等 | 机械臂抓放参数(高度/圆弧半径/夹爪) |
| `use_estimated_z` | 抓取 z 是否用深度估计值(默认 false,按桌面 z=0) |

## 设计笔记(踩坑记录,原 README 保留)

### 延迟
- 曾出现严重显示/控制延迟,根源是**键盘输入阻塞累积**:
  `get_key(0.1) → get_key(0.01)` 后延迟瞬间消失。主循环中所有轮询超时都要保持毫秒级。

### SAM 分割优化
- SAM 的 `set_image`/`predict` 是主要耗时点,**每帧只对置信度最高的目标执行一次**;
  其余检测框只保留 bbox(抓取只用到最优目标的掩码)。
- **阴影抑制**(本项目的核心小技巧):SAM 提示 = 检测框 + 采样点——
  - 中心区域采正点,锚定物体本体;
  - 框内四条边缘只采「比中心暗 30% 以上」的点作负点,抑制与物体相连的阴影,
    避免负点误伤本体;后处理再做填洞 + 最大连通域过滤。
  实现在 `detection_manager.py` 的 `sample_precise_points()`。

### 机械臂
- 夹爪 `finger_joint` 限位 [0, 0.8],0.8 为闭合;只能控开合角度,无力控。
- 长距离转移用**水平面内圆弧运动**(`move_in_arc_simple`),IK 解比自由空间更合理;
  失败自动回退直线插值。
- 轨迹点之间的 `time.sleep` 不能全去掉,否则会出现多段运动复合;改善方向见下。

## 已知待改进

- [ ] RGB/深度/TF 仍取"最新帧",未按 header.stamp 时间同步
- [ ] 抓取 z 默认取 0(桌面假设),`use_estimated_z: true` 可切换为深度估计值
- [ ] URDF 不含夹爪(finger_joint 只存在于 Isaac Sim 的 USD),MoveIt 监听 /joint_states 时
      会对 finger_joint 打 "not found in model" 告警——无害,但日志较吵;长远应在 URDF/SRDF 中补全夹爪
- [ ] 夹爪无力反馈,合爪宽度按物体类型经验估计(`_get_grasp_width`)
- [ ] 本机 Docker 的跨容器 DDS 数据不通,ROS 节点须与 Isaac Sim 分容器时需注意
      (move_group/converter/main.py 在同一容器内运行没有问题)

## 与 ../vision/ 的关系

两项目共享同一套 modules 设计;`vision/` 面向实物相机(SAM vit_b、抓取高度更低、
支持自定义抓取朝向),本项目面向 Isaac Sim(SAM vit_h、含 go_home 流程)。
修改公共逻辑时请两边同步。
