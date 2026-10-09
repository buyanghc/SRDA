# SRDA — reviewer edition

Code and final observations for **Single-Request Delegation Attacks (SRDA)**.
This lightweight edition contains only the paper's six model configurations and
reported experimental cohorts. It does not include pilot results or bulk logs.

## Start here

```bash
python3 review.py check                 # hashes, complete matrices and paired outcomes
python3 -m pip install -r requirements-analysis.txt
python3 review.py figures               # rebuild paper tables and five statistical figures
python3 review.py test                  # CPU-only release/launcher tests
```

Generated tables and figures appear in `paper/tables/` and `paper/figures/`.
The rebuild checks all 14 numerical tables against the frozen manuscript rows;
it does not run models or modify the paper.
Preparation checks are summarized in [VALIDATION.md](VALIDATION.md).

| Paper experiment | Included observations | Where to look |
|---|---:|---|
| RQ1–RQ2: guidance, feedback and topology | 68,040 | `results/controlled/` |
| RQ3: concurrent realistic workflows | 1,296 | `results/workflows/` |
| RQ4: attacks under three defenses | 9,720 | `results/defenses/` |

## Code guide

- `src/gatepath/`: shared frozen implementation, task registry and topology code.
- `src/variants/` and `configs/profiles/`: recorded model-specific differences.
- `experiments/`: paper-aligned configuration and fresh-run entrypoints.
- `analysis/`: result/table/figure generators; `results/`: final per-execution evidence.

Identical frozen files are stored once. Launchers reconstruct the selected exact
profile into ignored `.cache/runtimes/`; they do **not** merge different model
implementations. See [running experiments](docs/RUNNING.md) and the concise
[protocol mapping](docs/PROTOCOL.md).

## Evidence scope

Controlled metrics/configurations and defense metrics/lineage are losslessly compressed. Workflow
records retain every final execution and its numerical/protocol fields but omit
messages and action histories. Full trajectories and original source provenance
remain in the [pinned complete archive](https://github.com/buyanghc/SRDA/tree/da81aa0e2b620ddfa4b670f5f150dbbe729564b8).
No ERROR was removed or relabeled: RQ4 retains 234 exceptions separately from
completed-execution ASR. This edition is for lightweight inspection, not a
replacement for trajectory-level evidence.

Historical model-revision limitations remain documented in the run guide.
