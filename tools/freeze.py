"""Explicit checksum refresh, not called by check or experiment launchers."""
import hashlib
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
def main():
    names=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard'],cwd=ROOT,text=True).splitlines()
    rows=[]
    for name in sorted(set(names)):
        if name=='provenance/SHA256SUMS':continue
        p=ROOT/name
        if p.is_file():
            assert p.stat().st_size<8_000_000,name
            rows.append(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+name)
    (ROOT/'provenance/SHA256SUMS').write_text('\n'.join(rows)+'\n')
    print(f'Frozen {len(rows)} reviewer files')
if __name__=='__main__':main()
