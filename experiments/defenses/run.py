"""Fresh full RQ4 cell. No historical-run prerequisites or automatic retries."""
import argparse
import asyncio
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
from runtime import materialize
sys.path.insert(0,str(materialize('defenses')))
from defense_core import make_plan,run_one,summary
from io_utils import append,atomic_json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',required=True)
    p.add_argument('--seed',type=int,choices=[0,1,2],required=True)
    p.add_argument('--base-url',required=True)
    p.add_argument('--classifier-url',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--execute',action='store_true');p.add_argument('--limit',type=int)
    args=p.parse_args();args.local_stop=True
    cfg=json.loads(Path(__file__).with_name('config.json').read_text())
    cfg.update(cfg['seed_pairs'][args.seed]);cfg['seed_index']=args.seed
    models=json.loads(Path(__file__).with_name('agent-models.json').read_text());model=models[args.campaign]
    rows=make_plan(cfg,'formal');assert len(rows)==540
    if args.limit:rows=rows[:args.limit]
    if not args.execute:
        print(json.dumps({'planned':len(rows),'first':rows[0],'last':rows[-1]},indent=2));return
    if args.output.exists():raise SystemExit('Refuse to overwrite or silently resume an existing run')
    args.output.mkdir(parents=True)
    atomic_json(args.output/'manifest.json',dict(protocol=cfg['protocol'],config=cfg,model=model,phase='formal',campaign=args.campaign))
    atomic_json(args.output/'plan.json',rows)
    records={}
    for row in rows:
        append(args.output/'starts.jsonl',dict(key=row['key'],time=time.time()))
        started=time.monotonic();result=asyncio.run(run_one(row,cfg,model,args,args.output))
        result['controller_wall_seconds']=time.monotonic()-started
        append(args.output/'results.jsonl',result);records[row['key']]=result
        atomic_json(args.output/'summary.json',summary(rows,records))
        print(json.dumps({'recorded':len(records),'planned':len(rows),'status':result['status'],'key':row['key']}),flush=True)
        if result['status']=='ERROR':
            # Keep exact diagnostic evidence. Known model-output/capacity trial
            # outcomes may proceed; unknown failures stop the fresh run.
            from error_categories import classify
            event_file=args.output/'defense-audit.jsonl'
            events=[json.loads(x) for x in event_file.read_text().splitlines() if json.loads(x)['episode_key']==row['key']] if event_file.exists() else []
            category=classify(events)
            if category not in ('capacity','invalid_detector_output','truncated_detector_output'):
                raise RuntimeError('Unclassified error retained; inspect before continuation')
    atomic_json(args.output/'COVERAGE.json',dict(records=len(records),unique=len(records),clean=all(r['status']=='COMPLETED' for r in records.values())))

if __name__=='__main__':main()
