"""Single entrypoint for checking results, rebuilding figures, or planning reruns."""
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
COMMANDS={'check':'tools/verify.py','figures':'tools/rebuild.py',
          'plan':'experiments/run.py','serve':'experiments/serve.py','runtime':'tools/runtime.py'}

def main():
    if len(sys.argv)<2 or sys.argv[1] not in (*COMMANDS,'test'):
        print('Usage: python review.py {check|figures|plan|serve|runtime|test} [arguments]')
        print('check and test need no GPU; plan/serve do not run inference without --execute.')
        raise SystemExit(0 if len(sys.argv)<2 else 2)
    command=sys.argv[1]
    args=['-m','unittest','discover','-s',str(ROOT/'tests')] if command=='test' else [str(ROOT/COMMANDS[command])]
    subprocess.run([sys.executable,*args,*sys.argv[2:]],cwd=ROOT,check=True)

if __name__=='__main__':main()
