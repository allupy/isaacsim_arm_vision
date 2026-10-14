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
# 终端2: 启动 MoveIt
ros2 launch ur5_moveit_config start_moveit_isaac.launch.py
# 终端3: 本程序(在 ROS2 容器内,cwd = isaacsim_arm_vision)
python3 main.py
```

快捷键:`d` 检测并抓取最优目标 / `s` 拍照保存 / `c` 改类别 / `r` 归位 / `l` 历史 / `q` 退出。

## 目录结构

```
├── main.py                 主节点:订阅/显示/位姿估计/命令循环
├── config/
│   ├── config.py           DetectorConfig(YAML 加载 + 权重路径解析)
│   └── default_config.yaml 全部可调参数(模型/内参/手眼外参/机械臂)
├── modules/
│   ├── detection_manager.py  YOLO-World + SAM(仅最优目标分割)
│   ├── arm_manager.py        机械臂控制(IK/圆弧/直线/抓放流程)
│   ├── io_manager.py         结果保存与历史
│   ├── class_manager.py      开放词汇类别管理(伪自然语言交互)
│   └── keyboard_handler.py   非阻塞键盘输入
└── archive/                历史版本(废弃的 MoveItPy 方案、旧备份等)
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
- [ ] IK 逐点求解 + 服务调用串行,长轨迹可合并为单条 JointTrajectory(直线移动已如此)
- [ ] 夹爪无力反馈,合爪宽度按物体类型经验估计(`_get_grasp_width`)

## 与 ../vision/ 的关系

两项目共享同一套 modules 设计;`vision/` 面向实物相机(SAM vit_b、抓取高度更低、
支持自定义抓取朝向),本项目面向 Isaac Sim(SAM vit_h、含 go_home 流程)。
修改公共逻辑时请两边同步。
