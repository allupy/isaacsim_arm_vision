# 本次整理的检查范围

2026-10-28，Asia/Shanghai。

本页记录首次统一提交 `73470c0` 的验证。后续 main.py 检测入口修正、39 项检查和实际 ROS 图像回放见 [入口检查记录](MAIN_ENTRY_REVIEW.md)。

2026-10-29 的闭环交互入口改造与验证范围见 [交互说明](INTERACTIVE.md)。

2026-10-30 已通过 `main.py` 实际运行实时观察、苹果/杯子闭环搬运；杯子到区后侧倒，不能算直立放置通过。最新结果、录像和失败记录见 [端到端验证](MAIN_LIVE_VALIDATION_20261030.md)。

## 已通过

- `test_feedback.py`：23 项，反馈状态、观测配对、目标锁定与失败恢复等。
- `test_performance.py`：5 项，轨迹插值、时序、边界与策略配置等。
- `agent_workflow/test_workflow.py`：4 项，计划顺序、输入拒绝、LLM 校验和失败终止。
- `test_collaboration_config.py`：2 项，自定义容器执行，以及超时后只停止指定控制容器。
- 全部 Python 源文件语法编译；全部 shell 文件 `bash -n`。
- 13 个场景/物体文件 SHA-256 与尺寸检查。
- 初始 USD 在 ws 的 Isaac Sim 镜像中通过结构检查，包含 UR5、腕部相机和观察相机。独立 USD 依赖扫描发现 5 项材质引用，已记录在 `scenes/dependencies.json`；其中两项由 Isaac 提供，三项为历史在线 MDL 引用，本次未验证其在线解析或修复这些材质。
- 统一启动脚本的 Docker 参数检查：独立容器名、结果目录、只读源码挂载及自定义 ROS Domain。Docker 使用替身，未启动 GPU 仿真负载。
- 规则规划 dry-run。
- 4 个收录 MP4 全部逐帧解码成功，共 9291 帧；视频参数见 `docs/media/manifest.json`。
- 仓库主页相对文件链接、场景校验与 Git 差异空白检查。

## 本次没有重跑

本次完成源码/场景归档与协作入口整理，未重新执行完整 Isaac Sim 抓放，也未在空白系统重新安装 ROS/Isaac；未验证材质渲染。README 视频与数值来自已保存的历史实录和独立评分，不能视为当前提交重新测得的结果。

实体目的地仍是带标记的原型，杯架放置失败被保留在文档和评分里。固定区域结果不能外推到随机摆放、无标记承托物、任意语言任务或真机。

## 可复查命令

```bash
python -m unittest discover -s tests -p test_feedback.py
python -m unittest discover -s tests -p test_performance.py
python -m unittest discover -s tests -p test_collaboration_config.py
python agent_workflow/test_workflow.py
python scripts/check_assets.py
python agent_workflow/run.py --dry-run
bash -n scripts/arm.sh
git diff --check
```
