"""Compare the sum of individual gains with observed joint gain; review only."""
from pathlib import Path
from fractions import Fraction
import csv
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'combined-effects/source_data.csv'
with SOURCE.open(newline='') as f:
    source_rows = list(csv.DictReader(f))
cells = {(r['campaign'], r['condition']): r for r in source_rows}
assert len(cells) == 36
assert all(int(r['completed']) + int(r['errors']) == 1890 for r in source_rows)
MODELS = [
 ('qwen3-14b', 'Qwen3-14B'),
 ('qwen3.5-9B-nothinking', 'Qwen3.5-9B\n(no-thinking)'),
 ('qwen3.5-9B-thinking', 'Qwen3.5-9B\n(thinking)'),
 ('ministral3-14b', 'Ministral-3-14B'),
 ('gpt-oss-20b-low', 'GPT-OSS-20B'),
 ('llama3.1-8b', 'Llama-3.1-8B'),
]
def rate(model, condition):
    r = cells[model, condition]
    return Fraction(int(r['successes']), int(r['completed']) + int(r['errors']))

plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 15,
 'axes.labelsize': 17, 'xtick.labelsize': 13, 'ytick.labelsize': 14,
 'axes.linewidth': 1.1, 'axes.unicode_minus': False,
 'pdf.fonttype': 42, 'svg.fonttype': 'path', 'hatch.linewidth': .8})
fig, axes = plt.subplots(1, 2, figsize=(17, 6.4), sharey=True)
fig.subplots_adjust(left=.06, right=.99, bottom=.30, top=.92, wspace=.13)
centers = [3*i + 1.5 for i in range(len(MODELS))]
output = []
legends = []
for ax, regime, title in zip(axes, ['empty', 'natural'], ['(a) Empty feedback', '(b) Natural feedback']):
    additive, joint = [], []
    for model, _ in MODELS:
        baseline = rate(model, 'system-terminal')
        h = 100 * (rate(model, 'forward-terminal') - baseline)
        f = 100 * (rate(model, 'system-' + regime) - baseline)
        together = 100 * (rate(model, 'forward-' + regime) - baseline)
        additive.append(float(h + f)); joint.append(float(together))
        output.append(dict(campaign=model, feedback=regime,
            handoff_gain_pp=float(h), feedback_gain_pp=float(f),
            sum_individual_pp=float(h+f), actual_joint_pp=float(together),
            interaction_pp=float(together-h-f), exact_sum_pp=str(h+f),
            exact_joint_pp=str(together), exact_interaction_pp=str(together-h-f)))
    for j, (values, label, color, hatch) in enumerate([
        (additive, 'Sum of individual gains', '#376795', '/'),
        (joint, 'Actual joint gain', '#E76254', '\\')]):
        positions = [x+j-.5 for x in centers]
        ax.bar(positions, values, width=1, color=color, edgecolor='black',
               linewidth=.8, hatch=hatch, hatchcolor='white', label=label, zorder=3)
        for x, y in zip(positions, values):
            if abs(y) < 1:
                ax.annotate(f'{y:.2f}', (x,y), xytext=(0,4), textcoords='offset points',
                            ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_xlim(-1.65,19.65)
    ax.set_ylim(0,32)
    ax.set_yticks(range(0,31,5))
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='#E3E3E3', linewidth=.65)
    ax.set_xticks(centers, [label for _,label in MODELS], rotation=30,
                  ha='right', rotation_mode='anchor')
    ax.tick_params(axis='x', length=0, pad=8)
    ax.set_title(title, fontsize=17, pad=10)
    legends.append(ax.legend(loc='upper left', fontsize=12, frameon=True,
        fancybox=False, edgecolor='#BBBBBB', framealpha=1, handlelength=1.3,
        borderpad=.4))
axes[0].set_ylabel('ASR gain over baseline (pp)')
fig.canvas.draw()
for ax, legend in zip(axes, legends):
    box=legend.get_window_extent(fig.canvas.get_renderer())
    assert not any(box.overlaps(p.get_window_extent()) for p in ax.patches if p.get_height()>0)
assert all(centers[i+1]-centers[i]-2 == 1 for i in range(len(MODELS)-1))
assert sum(r['interaction_pp'] < 0 for r in output) == 3
assert next(r for r in output if r['campaign']=='gpt-oss-20b-low' and r['feedback']=='natural')['interaction_pp'] == 0
for ext in ['png','pdf','svg']:
    fig.savefig(ROOT/f'additive-versus-joint.{ext}', dpi=160)
with (ROOT/'additive-versus-joint-values.csv').open('w', newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(output[0])); w.writeheader(); w.writerows(output)
(ROOT/'additive-versus-joint-audit.json').write_text(json.dumps(dict(
    source=str(SOURCE), sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    formula_sum='100*((S_1T-S_0T)+(S_0f-S_0T))',
    formula_joint='100*(S_1f-S_0T)', denominator_per_cell=1890,
    exact_fraction_arithmetic=True, group_gap_in_bar_widths=1,
    legend_overlaps_bars=False, manuscript_updated=False,
    uncertainty='Descriptive pooled contrasts; no confidence intervals or significance claims.'),indent=2)+'\n')
