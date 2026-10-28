"""Validate the checked-in scene and object assets without Isaac Sim."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
entries = json.loads((root / 'scenes/manifest.json').read_text())
errors = []
for entry in entries:
    path = root / entry['path']
    if not path.is_file():
        errors.append(f"missing: {entry['path']}")
    elif path.stat().st_size != entry['size'] or hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
        errors.append(f"checksum mismatch: {entry['path']}")
if errors:
    raise SystemExit('\n'.join(errors))
print(f'Verified {len(entries)} scene/object files')
