import json,subprocess,tarfile
from pathlib import Path
root=Path('/home/aluupy/arm_feedback_experiments/20261027')
ids=['interactive_sequence_20261028_1','interactive_sequence_20261028_2','interactive_sequence_v2_20261028_1','interactive_sequence_v2_20261028_2','interactive_displace_20261028_1']
for identifier in ids:
    if identifier.startswith('interactive_sequence_20261028'):continue
    if not (root/'runs'/identifier/'result.json').exists():continue
    subprocess.run(['docker','exec','arm-feedback-control-20261027','python3','/experiment/src/tests/annotate_demo_recording.py','--id',identifier],check=True)
with tarfile.open(root/'interactive_demo_20261028.tgz','w:gz') as archive:
    for identifier in ids:
        if (root/'runs'/identifier).exists():
            archive.add(root/'runs'/identifier,arcname='runs/'+identifier)
            archive.add(root/'videos'/identifier,arcname='videos/'+identifier)
    archive.add(root/'runs/interactive_tests.log',arcname='interactive_tests.log')
print('packaged',flush=True)
