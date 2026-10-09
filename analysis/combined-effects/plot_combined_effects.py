"""Plot all five treatment-minus-baseline contrasts; no manuscript writes."""
from pathlib import Path
from fractions import Fraction
import csv
import hashlib
import json
import platform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import findfont
from matplotlib.text import Text
import numpy as np

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'main-asr/source-data.csv'
NEW = OLD
MODELS = [
    ('qwen3-14b', 'Qwen3-14B'),
    ('qwen3.5-9B-nothinking', 'Qwen3.5-9B\n(no-thinking)'),
    ('qwen3.5-9B-thinking', 'Qwen3.5-9B\n(thinking)'),
    ('ministral3-14b', 'Ministral-3-14B'),
    ('gpt-oss-20b-low', 'GPT-OSS-20B'),
    ('llama3.1-8b', 'Llama-3.1-8B'),
]
SERIES = [
    ('Handoff guidance', 'forward-terminal', '/'),
    ('Empty feedback', 'system-empty', '.'),
    ('Natural feedback', 'system-natural', '+'),
    ('Handoff guidance + Empty feedback', 'forward-empty', 'xx'),
    ('Handoff guidance + Natural feedback', 'forward-natural', '\\'),
]
PALETTES = {'requested': ['#376795', '#72BCD5', '#FFE6B7', '#FFD06F', '#E76254']}
CONDITIONS = ['system-terminal'] + [s[1] for s in SERIES]
cells = {}
with OLD.open(newline='') as f:
    for r in csv.DictReader(f):
        key=(r['campaign'],r['condition']); errors=int(r['raw_errors_counted_as_failures'])
        assert key not in cells
        cells[key]=dict(campaign=key[0],condition=key[1],completed=int(r['n'])-errors,
                       successes=int(r['successes']),errors=errors,source='analysis/main-asr/source-data.csv')
assert len(cells)==36
for campaign, _ in MODELS:
    for condition in CONDITIONS:
        c = cells[campaign, condition]
        assert c['completed'] + c['errors'] == 1890
        assert 0 <= c['successes'] <= c['completed']
assert sum(c['errors'] for c in cells.values()) == 5

values = np.zeros((5, len(MODELS)))
plot_rows = []
for j, (campaign, model) in enumerate(MODELS):
    baseline = cells[campaign, 'system-terminal']
    for i, (label, condition, _) in enumerate(SERIES):
        c = cells[campaign, condition]
        # User-approved evaluation rule: recorded ERROR episodes count as failures.
        evaluated = c['completed'] + c['errors']
        baseline_evaluated = baseline['completed'] + baseline['errors']
        delta = 100 * (Fraction(c['successes'], evaluated) -
                       Fraction(baseline['successes'], baseline_evaluated))
        values[i, j] = float(delta)
        plot_rows.append(dict(campaign=campaign, model=model.replace('\n', ' '),
            series=label, condition=condition, successes=c['successes'],
            completed=c['completed'], raw_errors_counted_as_failures=c['errors'],
            evaluated=evaluated, baseline_evaluated=baseline_evaluated,
            baseline_successes=baseline['successes'], baseline_completed=baseline['completed'],
            delta_pp=float(delta), exact_delta_pp=str(delta),
            denominator_policy='all_1890_episodes_errors_counted_as_failures'))
assert len(plot_rows) == 30
assert np.isclose(values[4, 4], 100 * (1226 - 1041) / 1890)
assert np.isclose(values[0, 0], 100 * 111 / 1890)
for name, rows in [('source_data.csv', list(cells.values())), ('effects.csv', plot_rows)]:
    with (ROOT / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

SCALE = 3
font_path = findfont('Times New Roman', fallback_to_default=True)
plt.rcParams.update({
    'font.family': 'Times New Roman', 'font.size': 27,
    'axes.labelsize': 29, 'xtick.labelsize': 24, 'ytick.labelsize': 24,
    'legend.fontsize': 25, 'axes.linewidth': 1.8, 'axes.unicode_minus': False,
    'svg.fonttype': 'path', 'pdf.fonttype': 42, 'hatch.linewidth': 1.1,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
})
audit = dict(sources={str(p.relative_to(ROOT.parents[1])): hashlib.sha256(p.read_bytes()).hexdigest() for p in (OLD, NEW)},
             source_cells=36, contrasts=30, total_errors=5,
             metric='treatment ASR minus system-terminal ASR, percentage points',
             denominator='1890 episodes per condition; raw ERROR episodes scored as failures by user instruction',
             interaction_difference_in_differences=False,
             uncertainty='No error bars; descriptive pooled proportions, no significance claims.',
             python=platform.python_version(), matplotlib=matplotlib.__version__,
             font=font_path, source_scale=SCALE, minimum_font_pt=24, variants={})
for palette_name, colors in PALETTES.items():
    fig, ax = plt.subplots(figsize=(6.5*SCALE, 3.15*SCALE))
    fig.subplots_adjust(left=.065, right=.993, bottom=.155, top=.87)
    width = 1.0
    # Keep the original 42-unit axis span and physical bar width.
    x = np.arange(len(MODELS)) * 6.0 + 3.0
    for i, ((label, _, hatch), color) in enumerate(zip(SERIES, colors)):
        positions = x + i - 2
        ax.bar(positions, values[i], width=width, color=color, edgecolor='black',
               linewidth=1.2, hatch=hatch, hatchcolor='white', label=label, zorder=3)
        for xpos, value in zip(positions, values[i]):
            if abs(value) < 1:
                ax.annotate(f'{value:.2f}', (xpos, max(value, 0)), xytext=(0, 5),
                    textcoords='offset points', ha='center', va='bottom',
                    rotation=90, fontsize=18, fontweight='bold', color='black', zorder=5)
    ax.set_xticks(x, [label for _, label in MODELS])
    ax.set_xlim(-3, 39)
    ax.set_ylim(0, 30)
    ax.set_yticks(np.arange(0, 31, 5))
    ax.set_ylabel('ASR change (pp)', labelpad=10)
    ax.tick_params(axis='x', length=0, pad=12)
    ax.tick_params(axis='y', length=6, width=1.5, pad=8)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
    handles, labels = ax.get_legend_handles_labels()
    # Column-major legend order: first row contains the three single factors.
    order = [0, 3, 1, 4, 2]
    legend = ax.legend([handles[i] for i in order], [labels[i] for i in order],
        loc='center', bbox_to_anchor=(.5, 1), ncol=3, frameon=True, fancybox=False,
        framealpha=1, facecolor='white', edgecolor='black', handlelength=1.3,
        columnspacing=.9, handletextpad=.45, borderpad=.35)
    legend.get_frame().set_linewidth(1.4)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    clipped = []
    for text in fig.findobj(match=Text):
        if not text.get_visible() or not text.get_text():
            continue
        box = text.get_window_extent(renderer)
        if not (fig.bbox.x0 <= box.x0 and box.x1 <= fig.bbox.x1 and
                fig.bbox.y0 <= box.y0 and box.y1 <= fig.bbox.y1):
            clipped.append(text.get_text())
    assert not clipped, clipped
    assert np.allclose(np.diff(x) - 5*width, width)
    legend_box = legend.get_window_extent(renderer)
    for patch in ax.patches:
        if patch.get_height() > 0:
            assert not legend_box.overlaps(patch.get_window_extent(renderer)), 'Legend overlaps bar'
    stem = f'palette-{palette_name}-six-configurations'
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(ROOT / f'{stem}.{ext}', dpi=100)
    # At 150 dpi this has the pixel dimensions of a 6.5-inch paper-width image.
    fig.savefig(ROOT / f'{stem}-preview.png', dpi=50)
    audit['variants'][palette_name] = dict(clipped_labels=clipped,
        bar_width_px=float(ax.transData.transform((1, 0))[0]-ax.transData.transform((0, 0))[0]),
        group_gap_in_bar_widths=1, within_group_gap=0, legend_overlaps_bars=False)
    plt.close(fig)
(ROOT / 'audit.json').write_text(json.dumps(audit, indent=2)+'\n')
for row in plot_rows:
    print(f"{row['model']:35s} {row['condition']:18s} {row['delta_pp']:8.4f} pp")
