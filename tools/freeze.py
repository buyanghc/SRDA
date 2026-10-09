"""Explicitly refresh release checksums; never called by verification or launchers."""
import hashlib
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]

def main():
    paths=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard'],cwd=ROOT,text=True).splitlines()
    paths=[p for p in sorted(set(paths)) if p!='provenance/SHA256SUMS' and (ROOT/p).is_file()]
    rows=[]
    for name in paths:
        p=ROOT/name
        assert p.stat().st_size<8_000_000,('Oversized file',name)
        with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
        rows.append(digest+'  '+name)
    (ROOT/'provenance/SHA256SUMS').write_text('\n'.join(rows)+'\n')
    print(f'Frozen {len(rows)} release files')

if __name__=='__main__':main()
