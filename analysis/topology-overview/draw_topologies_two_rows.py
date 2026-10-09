"""Editable topology atlas from saved graph data; no invented edges."""
from pathlib import Path
import json
import math
import html
import hashlib
import subprocess

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'topology-effects/topologies.json'
graphs={g['graph_name']:g for g in json.loads(SOURCE.read_text()) if g['seed']==0}
ORDER=[['G(1,2)','G(2,2)','G(3,2)','G(1,3)','G(2,3)','G(3,3)','G(1,6)'],
       ['G33-X2','G33-P2','G33-X3','G33-P3','GDAG','G16-ACYCLIC-CONTROL','G16-CYCLE']]
LABELS={
 'G(1,2)':('G(1,2)',), 'G(1,3)':('G(1,3)',), 'G(1,6)':('G(1,6)',),
 'G(2,2)':('G(2,2)',), 'G(2,3)':('G(2,3)',),
 'G(3,2)':('G(3,2)',), 'G(3,3)':('G(3,3)',),
 'G33-X2':('G(3,3) +','1 dead-end link'), 'G33-P2':('G(3,3) +','1 target-link'),
 'G33-X3':('G(3,3) +','2 dead-end links'), 'G33-P3':('G(3,3) +','2 target-links'),
 'GDAG':('DAG',), 'G16-ACYCLIC-CONTROL':('Chain + leaf',),
 'G16-CYCLE':('Chain + cycle',)}
assert len(graphs)==14 and set(sum(ORDER,[]))==set(graphs)
W,H=1820,790
svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
     '<rect width="100%" height="100%" fill="white"/>',
     '<defs><marker id="arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0 L7,3.5 L0,7 Z" fill="#74808D"/></marker><marker id="added-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0 L8,4 L0,8 Z" fill="#FFD06F"/></marker></defs>']
def text(x,y,label,size=20,fill='#20252B'):
    # Times New Roman uppercase cap height is approximately .662 em.
    # Use an explicit alphabetic baseline for E/T instead of renderer-dependent
    # dominant-baseline, which placed the enlarged capitals above the center.
    if label in ('E','T'):
        alignment=''
        y += size * .331 - .5
    else:
        alignment='dominant-baseline="middle"'
    svg.append(f'<text x="{x}" y="{y}" text-anchor="middle" {alignment} font-family="Times New Roman" font-size="{size}" fill="{fill}">{html.escape(label)}</text>')

def positions(g):
    name=g['graph_name']; pos={g['entry_agent']:(0,0)}
    if name=='GDAG':
        for n,p in zip(['Agent_005','Agent_003','Agent_002','Agent_001','Agent_006','Agent_004'],
                       [(-40,50),(40,50),(0,100),(-75,100),(75,100),(0,150)]):pos[n]=p
    else:
        base=graphs['G(3,3)'] if name.startswith('G33-') else g
        # Use the same backbone coordinates for all four matched variants.
        backbone=base['shortest_target_path']
        for level in range(1,max(int(n[7:9]) for n in g['nodes'] if n.startswith('Agent_L') and n[7:9].isdigit())+1):
            layer=[n for n in g['nodes'] if n.startswith(f'Agent_L{level:02d}_')]
            main=next(n for n in layer if n in backbone)
            pos[main]=(0,level*(36 if 'G16' in name or name=='G(1,6)' else 50))
            others=[n for n in layer if n!=main]
            for j,n in enumerate(others):pos[n]=((-52 if j==0 else 52),pos[main][1])
        if 'Agent_LOOP_FRESH' in g['nodes']:pos['Agent_LOOP_FRESH']=(67,108)
    assert set(pos)==set(g['nodes'])
    return pos

inventory=[]
for row,names in enumerate(ORDER):
    top=[100,445][row]
    cell=W/len(names)
    for col,name in enumerate(names):
        g=graphs[name]; pos=positions(g); cx=cell*(col+.5)
        svg.append(f'<g id="{g["graph_id"]}">')
        for a,b in g['edges']:
            added = name.startswith('G33-') and [a,b] not in graphs['G(3,3)']['edges']
            edge_color = '#FFD06F' if added else '#74808D'
            edge_marker = 'added-arrow' if added else 'arrow'
            edge_width = 4.2 if added else 1.6
            x1,y1=pos[a];x2,y2=pos[b]
            # Return edge is curved so both directions remain separately visible.
            if y2<y1:
                d=f'M {cx+x1+10},{top+y1-5} C {cx+x1+48},{top+y1+5} {cx+x2+48},{top+y2-5} {cx+x2+12},{top+y2+5}'
            else:
                dx,dy=x2-x1,y2-y1;l=math.hypot(dx,dy)
                r1=16 if a in [g['entry_agent'],g['target_agent']] else 10
                r2=18 if b==g['target_agent'] else 11
                d=f'M {cx+x1+dx/l*r1},{top+y1+dy/l*r1} L {cx+x2-dx/l*r2},{top+y2-dy/l*r2}'
            svg.append(f'<path data-source="{a}" data-target="{b}" d="{d}" fill="none" stroke="{edge_color}" stroke-width="{edge_width}" marker-end="url(#{edge_marker})"/>')
        for n,(x,y) in pos.items():
            entry=n==g['entry_agent'];target=n==g['target_agent']
            fill='#376795' if entry else '#E76254' if target else '#FFFFFF'
            stroke=fill if entry or target else '#74808D';r=15 if entry or target else 9
            svg.append(f'<circle data-node="{n}" cx="{cx+x}" cy="{top+y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="1.7"/>')
            if target:svg.append(f'<circle cx="{cx+x}" cy="{top+y}" r="18" fill="none" stroke="{stroke}" stroke-width="1.2"/>')
            if entry or target:text(cx+x,top+y+.5,'E' if entry else 'T',21,'white')
        for line,label in enumerate(LABELS[name]):
            text(cx,[374,722][row]+line*37,label,34)
        svg.append('</g>')
        inventory.append(dict(graph=name,nodes=len(pos),edges=len(g['edges'])))
# Shared key; no extra graph-level title or explanatory paragraphs in the canvas.
svg.append('<circle cx="430" cy="32" r="15" fill="#376795"/>')
text(430,33,'E',21,'white');text(546,33,'Entry agent',32)
svg.append('<circle cx="750" cy="32" r="15" fill="#E76254"/><circle cx="750" cy="32" r="18" fill="none" stroke="#E76254"/>')
text(750,33,'T',21,'white');text(877,33,'Target agent',32)
svg.append('<path d="M 1095,32 L 1150,32" stroke="#FFD06F" stroke-width="4.2" marker-end="url(#added-arrow)"/>')
text(1270,33,'Added edge',32)
svg.append('</svg>')
(ROOT/'topology-overview-two-rows.svg').write_text('\n'.join(svg))
for kind in ['png','pdf']:
    subprocess.run(['rsvg-convert','-f',kind,'-o',str(ROOT/f'topology-overview-two-rows.{kind}'),str(ROOT/'topology-overview-two-rows.svg')],check=True)
(ROOT/'audit-two-rows.json').write_text(json.dumps(dict(source=str(SOURCE),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    labels=LABELS,
    seed=0,graphs=inventory,isolated_nodes_preserved=True,edges_are_handoffs_not_feedback=True,
    human_approval='pending'),indent=2)+'\n')
