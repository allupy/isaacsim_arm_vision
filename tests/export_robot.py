from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.moveit_controller import build_ur5_moveit_config
cfg = build_ur5_moveit_config()
out = Path('/experiment/diagnostics')
out.mkdir(exist_ok=True)
(out/'robot.urdf').write_text(cfg['robot_description'])
print('EXPORTED',len(cfg['robot_description']))
