"""Portable launcher for the published matrices; planning is the default."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from runtime import materialize
MODELS = json.loads((ROOT/'experiments/models.json').read_text())
CONDITIONS = [h+'-'+f for h in ('system','forward') for f in ('terminal','empty','natural')]
DEFENSE_IDS = {'qwen3.5-9B-nothinking':'qwen35-nothinking','qwen3.5-9B-thinking':'qwen35-thinking',
               'llama3.1-8b':'llama31-8b'}

def command(args):
    profile=MODELS[args.model];runtime=materialize(args.model);settings=profile['settings']
    output=args.output.resolve(); exp=args.study+'-'+args.model+'-'+args.condition+'-s'+str(args.seed)
    if args.study=='controlled':
        if args.condition not in CONDITIONS:raise ValueError('Unknown controlled condition')
        cmd=[sys.executable,str(runtime/'scripts/run_request_response_batch.py'),
             '--exp-id',exp,'--formal-config',str(runtime/'configs'/('request_response_final_'+args.condition.replace('-','_')+'_full.yaml')),
             '--suite','all','--all-graphs','--repetitions','3','--run-root',str(output),
             '--base-url',args.base_url,'--model',profile['model']]
        for k,v in settings.items():
            if v is not None:cmd+=['--'+k.replace('_','-'),v]
        if args.execute:cmd+=['--execute','--continue-on-error']
        # A downloaded source ZIP has no Git metadata. Scientific files are
        # verified using SHA256SUMS; this flag only bypasses the old Git check.
        if args.execute:cmd+=['--allow-dirty']
        if args.limit:cmd+=['--max-episodes',str(args.limit)]
        return cmd,68040//36
    if args.study=='workflows':
        if args.condition not in ('system-terminal','system-natural','forward-terminal','forward-natural'):
            raise ValueError('RQ3 has only four concurrent conditions')
        config=ROOT/'experiments/workflows/configs'/args.model/(args.condition+'.json')
        native=[str(runtime/'scripts/run_mixed_workflow_batch.py')] if args.model=='qwen3-14b' else [
            str(ROOT/'experiments/workflows/run_cell.py'),'--source',str(runtime),'--campaign',args.model,'--kind','mixed']
        cmd=[sys.executable,*native,'--exp-id',exp,'--config',str(config),'--run-root',str(output),'--base-url',args.base_url]
        if args.execute:cmd+=['--execute','--continue-on-error']
        if args.limit:cmd+=['--max-episodes',str(args.limit)]
        return cmd,54
    cmd=[sys.executable,str(ROOT/'experiments/defenses/run.py'),'--campaign',DEFENSE_IDS.get(args.model,args.model),
         '--seed',str(args.seed),'--base-url',args.base_url,'--classifier-url',args.classifier_url,'--output',str(output/exp)]
    if args.execute:cmd+=['--execute']
    if args.limit:cmd+=['--limit',str(args.limit)]
    return cmd,540

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('study',choices=['controlled','workflows','defenses'])
    p.add_argument('--model',choices=list(MODELS),required=True)
    p.add_argument('--condition',default='forward-natural',choices=CONDITIONS)
    p.add_argument('--seed',type=int,choices=[0,1,2],default=0,help='Defense seed; controlled/workflows include all three seeds')
    p.add_argument('--base-url',default='http://127.0.0.1:8010/v1')
    p.add_argument('--classifier-url',default='http://127.0.0.1:9010')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/reruns')
    p.add_argument('--limit',type=int,help='Diagnostic subset only, never the published matrix')
    p.add_argument('--execute',action='store_true')
    args=p.parse_args()
    if args.limit is not None and args.limit<1:p.error('--limit must be positive')
    if args.study!='defenses' and args.seed!=0:p.error('Controlled/workflow cells already include all three seeds; do not supply --seed')
    if args.study=='defenses' and args.condition!='forward-natural':p.error('RQ4 is fixed to guidance + Natural feedback')
    cmd,n=command(args)
    print(json.dumps({'command':cmd,'full_cell_size':n,'execute':args.execute,
                      'diagnostic_subset':args.limit is not None},indent=2),flush=True)
    if not args.execute:return
    if (args.output.resolve()/ (args.study+'-'+args.model+'-'+args.condition+'-s'+str(args.seed))).exists():
        raise SystemExit('Output already exists; inspect it rather than silently replay records')
    subprocess.run([sys.executable,str(ROOT/'tools/verify.py'),'--sources-only'],check=True)
    env=dict(os.environ,REALISM_SOURCE_REVISION=json.loads((materialize(args.model)/'SOURCE.json').read_text())['source_revision'] or 'MIRROR',
             REALISM_ADAPTER_REVISION='artifact-v1',PYTHONDONTWRITEBYTECODE='1')
    subprocess.run(cmd,check=True,cwd=materialize(args.model),env=env)

if __name__=='__main__':main()
