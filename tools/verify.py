"""Check release hashes and published matrix consistency without third-party packages."""
import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import gzip
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def read(path):
    p=ROOT/path
    if p.suffix=='.gz':return json.loads(gzip.decompress(p.read_bytes()))
    return json.loads(p.read_text())
def csvrows(path):
    p=ROOT/path
    with (gzip.open(p,'rt',newline='') if p.suffix=='.gz' else p.open(newline='')) as f:
        return list(csv.DictReader(f))
def records(parts):
    for part in parts:
        p=ROOT/part['path'];raw=p.read_bytes()
        assert hashlib.sha256(raw).hexdigest()==part['sha256'],str(p)
        if p.suffix=='.gz':
            raw=gzip.decompress(raw)
            if 'uncompressed_sha256' in part:assert hashlib.sha256(raw).hexdigest()==part['uncompressed_sha256']
        rows=[json.loads(x) for x in raw.splitlines()]
        assert len(rows)==part['records']
        yield from rows

def check_hashes(sources_only=False):
    count=0
    for line in (ROOT/'provenance/SHA256SUMS').read_text().splitlines():
        digest,rel=line.split('  ',1)
        if sources_only and not rel.startswith(('src/','configs/','experiments/','tools/')):continue
        p=ROOT/rel
        assert p.is_file(),rel
        with p.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
        assert actual==digest,('Changed release file',rel)
        assert p.stat().st_size<8_000_000,('Oversized mirror file',rel)
        count+=1
    return count

def controlled():
    index=read('results/controlled/index.json');counts=csvrows('analysis/main-asr/source-data.csv')
    lookup={(r['campaign'],r['condition']):r for r in counts};keys=set();reference=None;errors=0
    assert len(lookup)==len(counts)==36
    for source in index['sources']:
        rows=csvrows(source['source']);model=source['campaign'];c=source['condition']
        assert hashlib.sha256(gzip.decompress((ROOT/source['source']).read_bytes())).hexdigest()==source['uncompressed_sha256']
        rr={r['episode_key'] for r in rows};assert len(rows)==len(rr)==1890
        if reference is None:reference=rr
        assert reference==rr
        graph_counts=Counter(r['graph_name'] for r in rows);assert len(graph_counts)==14 and set(graph_counts.values())=={135}
        assert Counter(r['repetition_id'] for r in rows)=={'0':630,'1':630,'2':630}
        successes=0;nerr=0
        for r in rows:
            key=(model,c,r['episode_key']);assert key not in keys;keys.add(key)
            assert r['status'] in ('COMPLETED','ERROR')
            success=r['status']=='COMPLETED' and r['attack_success']=='True'
            if success:assert r['target_tool_dispatched']==r['target_effect_verified']=='True'
            successes+=success;nerr+=r['status']=='ERROR'
            assert (int(r['graph_seed']),int(r['role_seed']))==(int(r['repetition_id']),100+int(r['repetition_id']))
        ref=lookup[model,c];assert successes==int(ref['successes']) and nerr==int(ref['raw_errors_counted_as_failures'])
        assert f'{100*successes/1890:.2f}'==ref['asr_percent'];errors+=nerr
        cfg=read(str(Path(source['source']).parent.parent/'config.json.gz'))
        assert cfg['budget']=={'max_messages':16,'max_runtime_seconds':90,'max_tool_iterations':4}
    assert len(keys)==68040 and errors==5
    return {'episodes':len(keys),'errors':errors,'cells':36}

def workflows():
    index=read('results/workflows/index.json');counts=csvrows('analysis/realistic-workflows/counts.csv')
    profiles=read('experiments/models.json');name_by_id=dict(zip(profiles,['Qwen3-14B','Qwen3.5-9B (no-thinking)','Qwen3.5-9B (thinking)','Ministral-3-14B-Instruct','GPT-OSS-20B','Llama-3.1-8B']))
    lookup={(r['model'],r['condition']):r for r in counts};keys=set()
    for cell in index['cells']:
        rows=list(records(cell['parts']));assert len(rows)==54
        model=cell['model'];condition=cell['condition'];ref=lookup[name_by_id[model],condition]
        totals=Counter()
        for r in rows:
            key=(model,condition,r['episode_key']);assert key not in keys;keys.add(key)
            assert r['status']=='COMPLETED' and r['condition']=='normal_plus_attack'
            p=r['report'];assert p['joint_success']==(p['attack_success'] and p['benign_workflow_success'])
            assert p['shared_environment'] and p['execution_mode']=='concurrent_requests_shared_environment'
            for kind in ('normal','attack'):
                budget=p[kind+'_request_report']['budget'];assert budget=={'max_messages':64,'max_runtime_seconds':180,'max_tool_iterations':4}
            for k in ('attack_success','benign_workflow_success','joint_success'):totals[k]+=p[k]
        assert totals['attack_success']==int(ref['attack_successes'])
        assert totals['benign_workflow_success']==int(ref['benign_successes'])
        assert totals['joint_success']==int(ref['joint_successes'])
    assert len(keys)==1296 and len(index['cells'])==24
    return {'episodes':len(keys),'cells':24,'errors':0}

def defenses():
    index=read('results/defenses/index.json');metrics=list(records(index['metrics']));lookup={(r['campaign'],r['key']):r for r in metrics}
    assert len(lookup)==len(metrics)==9720
    assert Counter(r['status'] for r in metrics)=={'COMPLETED':9486,'ERROR':234}
    pairs=defaultdict(set);counts=defaultdict(Counter)
    for r in metrics:
        pairs[r['campaign'],r['instance_id'],r['graph'],r['graph_seed'],r['role_seed']].add(r['defense'])
        counts[r['defense']]['planned']+=1
        if r['status']=='COMPLETED':counts[r['defense']]['valid']+=1;counts[r['defense']]['successes']+=r['attack_success']
        else:assert r['attack_success'] is None
    assert len(pairs)==2430 and all(v=={'none','protectai','promptguard','llm_detector'} for v in pairs.values())
    expected=read('results/defenses/published-summary.json')['groups']
    for arm,c in counts.items():
        assert c['planned']==expected[arm]['planned'] and c['valid']==expected[arm]['valid']
        assert c['successes']==expected[arm]['attack_successes_valid']
    published=read('results/defenses/published-summary.json')
    for arm in ['protectai','promptguard','llm_detector']:
        matched=[]
        for r in metrics:
            if r['defense']!=arm:continue
            no_key='none__'+r['key'].split('__',1)[1]
            baseline=lookup[r['campaign'],no_key]
            if r['status']==baseline['status']=='COMPLETED':matched.append((baseline,r))
        ref=published['paired_cost_and_success'][arm]
        assert len(matched)==ref['valid_pairs']
        assert sum(a['attack_success'] for a,b in matched)==ref['none_successes']
        assert sum(b['attack_success'] for a,b in matched)==ref['defended_successes']
    lineage=list(records(index['lineage']))
    assert len(lineage)==len({(r['campaign'],r['key']) for r in lineage})==9720
    assert {(r['campaign'],r['key']) for r in lineage}==set(lookup)
    return {'episodes':9720,'completed':9486,'errors':234,'four_arm_pairs':2430,
            'evidence_scope':'Final metrics and canonical lineage; bulk trajectories are in the pinned complete archive'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--sources-only',action='store_true');args=p.parse_args()
    report={'hashes_checked':check_hashes(args.sources_only)}
    if not args.sources_only:report.update(controlled=controlled(),workflows=workflows(),defenses=defenses())
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
