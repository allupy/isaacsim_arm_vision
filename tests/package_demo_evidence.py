import tarfile
from pathlib import Path
root=Path('/home/aluupy/arm_feedback_experiments/20261027')
with tarfile.open(root/'yolo_everyday_demo_evidence_20261028.tgz','w:gz') as archive:
    for name in ['yolo_apple_20261028_1','yolo_banana_v4_20261028_1','yolo_cup_observer_20261028_1']:
        archive.add(root/'videos'/name,arcname='videos/'+name)
    for folder in root.glob('runs/yolo_*20261028_1'):
        if folder.is_dir():archive.add(folder,arcname='runs/'+folder.name)
    for file in root.glob('diagnostics/*demo*'):
        if file.is_file():archive.add(file,arcname='diagnostics/'+file.name)
print('packaged',flush=True)
