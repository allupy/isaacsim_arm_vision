"""Offline FK comparison of the actual MoveIt URDF and authored USD link poses."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]/'results'
scene = json.loads((ROOT/'static_scene.json').read_text())
robot = ET.parse(ROOT/'robot.urdf').getroot()
joints = {j.find('child').get('link'):j for j in robot.findall('joint')}
angles = {name:np.deg2rad(float(data['attributes'].get('state:angular:physics:position','0')))
          for name,data in scene['joints'].items()}

def fk(link, positions):
    if link not in joints:
        return np.eye(4)
    joint = joints[link]
    origin = joint.find('origin')
    xyz = [float(v) for v in origin.get('xyz','0 0 0').split()] if origin is not None else [0.,0.,0.]
    rpy = [float(v) for v in origin.get('rpy','0 0 0').split()] if origin is not None else [0.,0.,0.]
    transform = np.eye(4)
    transform[:3,3] = xyz
    transform[:3,:3] = Rotation.from_euler('xyz',rpy).as_matrix()
    if joint.get('type') in ['revolute','continuous']:
        axis = np.array([float(v) for v in joint.find('axis').get('xyz').split()])
        rotate = np.eye(4)
        rotate[:3,:3] = Rotation.from_rotvec(axis*positions.get(joint.get('name'),0.)).as_matrix()
        transform = transform @ rotate
    return fk(joint.find('parent').get('link'),positions) @ transform

rows = []
for name,data in scene['links'].items():
    usd = np.array(data['world_matrix_row_major']).T
    predicted = fk(name,angles)
    # Use base_link as the common reference even if the URDF world offset changes.
    predicted = np.linalg.inv(fk('base_link',angles)) @ predicted
    usd = np.linalg.inv(np.array(scene['links']['base_link']['world_matrix_row_major']).T) @ usd
    rows.append(dict(link=name, usd=usd[:3,3].tolist(), urdf=predicted[:3,3].tolist(),
                     position_error_m=float(np.linalg.norm(usd[:3,3]-predicted[:3,3])),
                     rotation_error_deg=float(np.rad2deg(Rotation.from_matrix(usd[:3,:3].T@predicted[:3,:3]).magnitude()))))
output = {'authored_angles_rad':angles,'links':rows}
(ROOT/'kinematic_comparison.json').write_text(json.dumps(output,indent=2))
print(json.dumps(output,indent=2))
