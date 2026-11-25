"""Unpack and verify the versioned, byte-preserved AIMS input snapshot."""
from pathlib import Path
from zipfile import ZipFile
import hashlib
import json

def prepare():
    base=Path(__file__).resolve().parent.parent/'data/aims_local_stock'
    manifest=json.loads((base/'manifest.json').read_text(encoding='utf-8'))['files']
    with ZipFile(base/'ecology_snapshot_20261120.zip') as archive:
        if set(archive.namelist()) != set(manifest):
            raise ValueError('Archive members differ from the snapshot manifest')
        for name, expected in manifest.items():
            target=(base/name).resolve()
            if not target.is_relative_to(base.resolve()):
                raise ValueError('Snapshot member is outside the data directory')
            if not target.exists():
                data=archive.read(name)
                if hashlib.sha256(data).hexdigest()!=expected['sha256']:
                    raise ValueError('Archive checksum mismatch: '+name)
                target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(data)
            if hashlib.sha256(target.read_bytes()).hexdigest()!=expected['sha256']:
                raise ValueError('Existing input differs from the published snapshot: '+name)
    print(f'Verified {len(manifest)} snapshot files')

if __name__=='__main__':
    prepare()
