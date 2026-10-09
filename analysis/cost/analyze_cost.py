"""Audit recorded full-execution costs for the current six-configuration cohort.

Read-only experiment sources; writes only derived CSV/JSON beside this script.
No monetary, GPU-energy, or time-to-first-success interpretation is intended.
"""
import csv
import gzip
import hashlib
import json
import statistics as stats
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONDITIONS = [f'{h}-{f}' for h in ('system', 'forward')
              for f in ('terminal', 'empty', 'natural')]


def read_csv(path):
    with (gzip.open(path,'rt',newline='') if path.suffix=='.gz' else path.open(newline='')) as stream:
        return list(csv.DictReader(stream))


def write_csv(name, rows):
    with (HERE / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean(rows, field):
    values = [float(r[field]) for r in rows if r.get(field) not in ('', None)]
    return stats.mean(values) if values else None


def main():
    evidence = json.loads((ROOT / 'analysis/main-asr/verification.json').read_text())
    sources = [(s['campaign'],s['condition'],ROOT / s['source']) for s in evidence['sources']]
    ledger = {(r['campaign'], r['condition']): r for r in
              read_csv(ROOT / 'analysis/combined-effects/source_data.csv')}
    common_keys = None
    all_rows, summary, manifest = {}, [], []
    for campaign, condition, path in sources:
        rows = read_csv(path)
        keys = {r['episode_key'] for r in rows}
        assert len(rows) == len(keys) == 1890
        if common_keys is None:
            common_keys = keys
        assert common_keys == keys
        successes = sum(r['attack_success'] == 'True' for r in rows)
        errors = sum(r['status'] == 'ERROR' for r in rows)
        reference = ledger[campaign, condition]
        assert successes == int(reference['successes'])
        assert errors == int(reference['errors'])
        config = json.loads(gzip.decompress((path.parent.parent / 'config.json.gz').read_bytes()))
        assert config['budget'] == dict(max_messages=16, max_runtime_seconds=90.0,
                                        max_tool_iterations=4)
        tokens = [float(r['total_tokens']) for r in rows if r['total_tokens'] != '']
        assert all(t > 0 for t in tokens)
        missing = [r['episode_key'] for r in rows if r['total_tokens'] == '']
        assert all(r['status'] == 'ERROR' for r in rows if r['total_tokens'] == '')
        record = dict(campaign=campaign, condition=condition, n=len(rows),
                      successes=successes, asr_percent=100*successes/len(rows),
                      errors=errors, token_observed_n=len(tokens),
                      token_missing_n=len(missing), mean_tokens=stats.mean(tokens),
                      median_tokens=stats.median(tokens), recorded_total_tokens=sum(tokens),
                      recorded_tokens_per_success=sum(tokens)/successes,
                      mean_messages=mean(rows, 'messages_sent'),
                      mean_elapsed_seconds=mean(rows, 'elapsed_seconds'),
                      message_budget_exhausted=sum(r['message_budget_exhausted']=='True' for r in rows),
                      runtime_budget_exhausted=sum(r['runtime_budget_exhausted']=='True' for r in rows),
                      mean_repeated_node_visits=mean(rows, 'repeated_node_visit_count'),
                      mean_tokens_success=mean([r for r in rows if r['attack_success']=='True'], 'total_tokens'),
                      mean_tokens_unsuccessful=mean([r for r in rows if r['attack_success']!='True'], 'total_tokens'))
        summary.append(record)
        all_rows[campaign, condition] = rows
        manifest.append(dict(campaign=campaign, condition=condition, source=str(path.relative_to(ROOT)),
                             sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             missing_token_episode_keys=missing))
    lookup = {(r['campaign'], r['condition']): r for r in summary}
    comparisons = []
    for campaign in dict.fromkeys(r['campaign'] for r in summary):
        for before, after in [('system-terminal','forward-terminal'),
                              ('system-terminal','forward-natural'),
                              ('forward-terminal','forward-natural'),
                              ('forward-empty','forward-natural')]:
            a, b = lookup[campaign, before], lookup[campaign, after]
            ar = {r['episode_key']: r for r in all_rows[campaign, before]}
            br = {r['episode_key']: r for r in all_rows[campaign, after]}
            paired = [k for k in ar if ar[k]['total_tokens'] and br[k]['total_tokens']]
            comparisons.append(dict(campaign=campaign, before=before, after=after,
                                    asr_change_pp=b['asr_percent']-a['asr_percent'],
                                    mean_token_ratio=b['mean_tokens']/a['mean_tokens'],
                                    mean_token_change_percent=100*(b['mean_tokens']/a['mean_tokens']-1),
                                    mean_messages_before=a['mean_messages'],
                                    mean_messages_after=b['mean_messages'],
                                    recorded_tokens_per_success_ratio=b['recorded_tokens_per_success']/a['recorded_tokens_per_success'],
                                    cost_matched_n=len(paired),
                                    paired_mean_token_delta=stats.mean(float(br[k]['total_tokens'])-float(ar[k]['total_tokens']) for k in paired)))
    assert len(summary) == 36 and sum(r['n'] for r in summary) == 68040
    write_csv('condition-summary.csv', summary)
    write_csv('condition-comparisons.csv', comparisons)
    (HERE / 'audit.json').write_text(json.dumps(dict(
        episodes=68040, cost_observed=68035, missing_cost=5,
        scope='Current six-configuration main-experiment cohort only; no pilot or realistic-workflow runs.',
        denominator='ASR uses all 1890 attempts per cell. Token/message means use observed records, including unsuccessful and budget-exhausted executions. Missing error costs are not zero.',
        cost_semantics='Recorded total_tokens = prompt + completion over full execution, including any activity after first verified success. Not minimum attack cost or money. Messages count direct requests, not returned feedback or model calls.',
        efficiency_semantics='Recorded total tokens / all verified successes includes failed-attempt costs; Llama Natural values are lower bounds due to missing error costs.',
        sources=manifest), indent=2)+'\n')
    for row in comparisons:
        print(row['campaign'], row['before'], '->', row['after'],
              f"ASR {row['asr_change_pp']:+.2f} pp; tokens {row['mean_token_ratio']:.3f}x; "
              f"tokens/success {row['recorded_tokens_per_success_ratio']:.3f}x")


if __name__ == '__main__':
    main()
