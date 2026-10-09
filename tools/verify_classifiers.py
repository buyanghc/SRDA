"""Verify locally downloaded pinned classifier snapshots; never downloads weights."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def verify(folder,spec):
    for name,expected in spec['files'].items():
        p=folder/name
        if not p.is_file():raise ValueError(f'Missing classifier file: {name}')
        size=p.stat().st_size
        if size!=expected['size']:raise ValueError(f'Wrong file size: {name}')
        if expected['sha256']:
            with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
            if digest!=expected['sha256']:raise ValueError(f'SHA256 mismatch: {name}')
        else:
            digest=hashlib.sha1(f'blob {size}\0'.encode()+p.read_bytes()).hexdigest()
            if digest!=expected['git_blob_sha1']:raise ValueError(f'Git blob mismatch: {name}')
    return len(spec['files'])

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',type=Path,required=True)
    p.add_argument('--write',action='store_true',help='Write VERIFIED.json only after all files pass')
    a=p.parse_args();manifest=json.loads((ROOT/'experiments/defenses/models.json').read_text())
    results={}
    for arm,spec in manifest.items():
        folder=a.models/arm/spec['revision'];results[arm]=verify(folder,spec)
    if a.write:
        for arm,spec in manifest.items():
            (a.models/arm/spec['revision']/'VERIFIED.json').write_text(json.dumps(spec,indent=2)+'\n')
    print(json.dumps(dict(verified_files=results,wrote_receipts=a.write),indent=2))

if __name__=='__main__':main()
