"""Rebuild published numerical tables and active plots from released observations."""
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from verify import records,read

def numerical_rows(path):
    return [re.findall(r'(?<![A-Za-z])[-+]?\d+(?:\.\d+)?',line) for line in path.read_text().splitlines()
            if ' & ' in line and re.search(r' & [-+]?\d',line)]

def defense_table():
    metrics=list(records(read('results/defenses/index.json')['metrics']));cells=defaultdict(Counter)
    for r in metrics:
        for graph in (r['graph'],'Overall'):
            c=cells[r['defense'],graph];c['planned']+=1
            if r['status']=='COMPLETED':c['completed']+=1;c['successes']+=r['attack_success']
            else:c['errors']+=1
    labels=[('none','None'),('protectai','ProtectAI'),('promptguard','Prompt Guard'),('llm_detector','LLM Detector')]
    graphs=['G(1,3)','G(3,3)','G33-P3','Overall'];rows=[];counts=[]
    for arm,label in labels:
        values=[]
        for graph in graphs:
            c=cells[arm,graph];values.append(f"{100*c['successes']/c['completed']:.2f}")
            counts.append(dict(defense=arm,graph=graph,**{k:c[k] for k in ['planned','completed','errors','successes']}))
        rows.append(label+' & '+' & '.join(values)+r' \\')
    template=(ROOT/'paper/reference-tables/defense-asr.tex').read_text()
    for (_,label),row in zip(labels,rows):
        template=re.sub(r'^'+re.escape(label)+r' & .*$',lambda _:row,template,flags=re.M)
    (ROOT/'paper/tables/defense-asr.tex').write_text(template)
    (ROOT/'analysis/defenses').mkdir(exist_ok=True)
    with (ROOT/'analysis/defenses/counts.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=counts[0]);w.writeheader();w.writerows(counts)

def main():
    (ROOT/'paper/tables').mkdir(parents=True,exist_ok=True);(ROOT/'paper/figures').mkdir(exist_ok=True)
    scripts=['main-asr/build_table.py','combined-effects/plot_combined_effects.py',
             'cost/analyze_cost.py','cost/build_table.py','topology-effects/prepare_topology.py',
             'topology-effects/plot_topology_marginal.py','interaction-effects/plot_additive_comparison.py',
             'realistic-workflows/build_table.py','appendix-d/build_results.py','appendix-e/build_results.py']
    for script in scripts:
        subprocess.run([sys.executable,str(ROOT/'analysis'/script)],check=True,cwd=ROOT)
    defense_table()
    # Topology atlas source is included; its optional SVG-to-PDF conversion
    # uses a system rsvg-convert. Numerical figure scripts need no TeX engine.
    figures={'combined-effects/palette-requested-six-configurations.pdf':'combined-effects.pdf',
             'interaction-effects/additive-versus-joint.pdf':'interaction-additive-comparison.pdf'}
    for name in ['outdegree','depth','paths']:
        figures['topology-effects/'+name+'-marginal.pdf']='topology-'+name+'-effects.pdf'
    for src,dst in figures.items():shutil.copyfile(ROOT/'analysis'/src,ROOT/'paper/figures'/dst)
    checks={}
    for p in sorted((ROOT/'paper/tables').glob('*.tex')):
        reference=ROOT/'paper/reference-tables'/p.name
        if reference.exists():
            assert numerical_rows(p)==numerical_rows(reference),('Numerical table differs from current manuscript',p.name)
            checks[p.name]='numerical rows match'
    print(json.dumps({'tables':checks,'figures':list(figures.values()),'output':'paper/'},indent=2))

if __name__=='__main__':main()
