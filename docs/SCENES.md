# 场景与资产

## 已收录

- `scenes/workbench/workbench.usd`：ws 上实际使用的工作台初始场景快照（UR5、桌面、相机、原夹爪、三色方块）。
- `sim_scene/build_workbench.py`：历史场景构建脚本，需要额外的 `ur5_depth.usd` 原始资产；运行已收录场景无需重建。
- `modules/short_parallel_gripper.py`：在运行时构建短平行夹爪，不覆盖原始 USD。
- `modules/demo_scene.py`：将方块替换成缩放后的 YCB 苹果、香蕉、杯子，包含物理碰撞。
- `modules/destination_scene.py`：构建盘子和杯架原型，生成 ArUco 标记。
- `assets/ycb/`：上述三个物体运行所需的 OBJ、MTL 和纹理；去掉了重复压缩包及未使用的 DAE/STL/PLY。
- `ros2/ur5_moveit_config/`：当前工作站的 MoveIt 配置与启动代码快照；MoveIt 本体、ROS 驱动和编译产物仍由外部工作空间提供。

场景配置保存在 `sim_scene/runtime_*.json`，可选 `optimized`（方块）、`short_parallel`（短夹爪方块）、`everyday_demo`（日常物体）、`destinations`（带标记承托物）。

## 来源与坐标约定

YCB 模型来自 YCB Object and Model Set / Google 16k，模型编号为 011_banana、013_apple、025_mug；来源 https://www.ycbbenchmarks.com/ 。第三方资产保留其原有权利，仓库没有为其重新授权。UR5/Robotiq/相机模型同样属于第三方模型。

`workbench.usd` 是原工作站场景的快照，USD 标称 metersPerUnit 与场景坐标存在历史差异。直接桥接脚本按 metersPerUnit 修正深度，并在内存副本修复相机安装和初始关节姿态。不要直接把这一仿真约定用于真机。

SHA-256 与文件尺寸见 `scenes/manifest.json`。启动前使用 `python scripts/check_assets.py` 验证。

## 材质依赖

原 USD 的几何已经包含在文件中，但材质并非完全离线。独立 USD 扫描报告 5 项 MDL 引用，详见 `scenes/dependencies.json`。`OmniPBR.mdl`、`OmniGlass.mdl` 由现有 Isaac 镜像的 `/isaac-sim/kit/mdl/core/Base/` 提供；其余 3 项是原 USD 中的旧版在线材质路径，包含历史 `http:/` 写法。本次保留原场景，没有替换或重建材质，也未验证这些在线材质的可用性。新机器可能出现材质缺失，需在 Isaac 材质编辑器中重新绑定；几何检查和文件哈希通过不代表渲染检查通过。
