"""Local-only topology/result inventory; does not change experiment or paper files."""
import ast
import csv
import hashlib
import json
import sys
import types
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
SOURCE = PROJECT / 'analysis/main-asr/episode-outcomes.csv'
COUNTS = PROJECT / 'analysis/combined-effects/source_data.csv'
GRAPH = PROJECT / 'runtimes/gpt-oss-20b-low/gatepath/experiment_graphs.py'
CONFIG = PROJECT / 'results/controlled/qwen3-14b/system-terminal/config.json'
CONDITIONS = ['system-terminal', 'system-empty', 'system-natural',
              'forward-terminal', 'forward-empty', 'forward-natural']

def read_csv(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))

def save_csv(name, rows):
    with (ROOT / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

sources = [SOURCE, COUNTS, GRAPH, CONFIG]
episodes = read_csv(SOURCE)
for r in episodes:
    r['success'] = int(r['success'])
assert len(episodes) == 68040
groups = defaultdict(list)
keys = defaultdict(set)
for r in episodes:
    groups[r['campaign'], r['condition'], r['graph']].append(r)
    key = (r['campaign'], r['condition'])
    assert r['episode_key'] not in keys[key]
    keys[key].add(r['episode_key'])
assert len(groups) == 504 and all(len(v) == 135 for v in groups.values())
assert len({frozenset(v) for v in keys.values()}) == 1
for r in read_csv(COUNTS):
    actual = sum(sum(x['success'] for x in v) for (m,c,g),v in groups.items()
                 if (m,c) == (r['campaign'],r['condition']))
    assert actual == int(r['successes']), (r['campaign'],r['condition'])
cell_rows = [dict(campaign=m, condition=c, graph=g, n=len(v),
                 successes=sum(r['success'] for r in v),
                 asr_pp=100*sum(r['success'] for r in v)/len(v))
             for (m,c,g),v in groups.items()]
save_csv('by_model_topology_condition.csv', cell_rows)
rates = {(r['campaign'],r['graph'],r['condition']):r['asr_pp'] for r in cell_rows}
models = list(dict.fromkeys(r['campaign'] for r in episodes))
graphs = json.loads(CONFIG.read_text())['graph_names']
effects = []
for m in models:
    for g in graphs:
        st,se,sn,ft,fe,fn = [rates[m,g,c] for c in CONDITIONS]
        effects.append(dict(campaign=m, graph=g, baseline_asr=st,
            guidance=ft-st, empty=se-st, natural=sn-st,
            interaction_empty=fe-ft-se+st, interaction_natural=fn-ft-sn+st))
save_csv('effects_by_model.csv', effects)
metrics = ['baseline_asr','guidance','empty','natural','interaction_empty','interaction_natural']
pooled = []
for g in graphs:
    row = dict(graph=g)
    for metric in metrics:
        vals = [r[metric] for r in effects if r['graph']==g]
        row[metric] = mean(vals)
        row[metric+'_positive_models'] = sum(v>1e-8 for v in vals)
        row[metric+'_negative_models'] = sum(v < -1e-8 for v in vals)
    pooled.append(row)
save_csv('effects_model_mean.csv',pooled)

# Load unchanged topology algorithms, omitting only the simulator world import.
# That import is used solely by to_world(), which is never called here.
tree = ast.parse(GRAPH.read_text())
tree.body = [n for n in tree.body if not (isinstance(n,ast.ImportFrom) and n.module=='world')]
module = types.ModuleType('_topology_inventory'); sys.modules[module.__name__] = module
exec(compile(tree,str(GRAPH),'exec'), module.__dict__)
structures, raw_graphs = [], []
for g in graphs:
    variants=[]
    for seed in [0,1,2]:
        t=module.build_named_topology(g, seed=seed)
        raw_graphs.append(t.as_dict())
        variants.append(dict(graph=g, nodes=t.node_count, edges=t.edge_count,
            entry_outdegree=len(t.contacts_for(t.entry_agent)), max_outdegree=t.max_out_degree,
            shortest_target_depth=len(t.shortest_path_to_target())-1,
            simple_target_paths=t.path_count_to_target(), has_cycle=t.has_cycle))
    assert variants[0]==variants[1]==variants[2]
    structures.append(variants[0])
save_csv('topology_structure.csv', structures)
(ROOT/'topologies.json').write_text(json.dumps(raw_graphs,indent=2)+'\n')
(ROOT/'audit.json').write_text(json.dumps(dict(episodes=len(episodes),cells=len(groups),
    executions_per_model_condition_topology=135, pooled_cells_match_table=True,
    matched_episode_keys=True, model_configs=len(models), topology_count=len(graphs),
    sources={str(p.relative_to(PROJECT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    topology_loading='Original module AST with unused relative world import removed; algorithms unchanged',
    path_metric='Simple directed entry-to-target paths; not observed effective paths',
    aggregation='Equal weight across six configurations, including both Qwen3.5 modes',
    uncertainty='Descriptive summaries, no significance tests'),indent=2)+'\n')
print('STRUCTURE')
for r in structures: print(r)
print('MEAN EFFECTS')
for r in pooled: print(r['graph'], ' '.join(f'{k}={r[k]:.2f} ({r[k+"_positive_models"]}+/ {r[k+"_negative_models"]}-)' for k in metrics[1:]))
