"""Reconstruct an exact frozen profile from shared files, without inference."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]

def materialize(profile):
    if profile not in {p.stem for p in (ROOT/'configs/profiles').glob('*.json')}:
        raise ValueError('Unknown frozen runtime profile')
    manifest = json.loads((ROOT/'configs/profiles'/(profile+'.json')).read_text())['files']
    cache = ROOT/'.cache/runtimes'
    target = cache/profile
    for rel, item in manifest.items():
        source = ROOT/item['source']
        if not source.resolve().is_relative_to((ROOT/'src').resolve()):
            raise ValueError('Source outside shared frozen directory')
        if Path(rel).is_absolute() or '..' in Path(rel).parts:
            raise ValueError('Unsafe profile destination')
        if hashlib.sha256(source.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('Changed frozen source: '+item['source'])
    if not target.exists():
        cache.mkdir(parents=True,exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=profile+'-',dir=cache))
        try:
            for rel, item in manifest.items():
                path=stage/rel;path.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(ROOT/item['source'],path)
            try:
                stage.rename(target)
            except OSError:
                if not target.is_dir():raise
        finally:
            if stage.exists():shutil.rmtree(stage) # only this generated staging directory
    for rel, item in manifest.items():
        if hashlib.sha256((target/rel).read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError('Changed materialized runtime: '+profile+'/'+rel)
    return target

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',required=True)
    args=p.parse_args();print(materialize(args.profile))

if __name__=='__main__':main()
