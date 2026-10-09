# SRDA reproducibility artifact

This project accompanies the manuscript's current six-configuration evaluation.
It contains only the published study matrices, canonical results, relevant runtime
sources and analysis entrypoints. It does not contain model weights, credentials,
institutional scheduler/account settings, pilot results or obsolete experimental
branches. Original preservation archives remain outside this release.

## Two reproduction paths

### 1. Verify results and rebuild paper outputs (no GPU)

```bash
python3 tools/verify.py
python3 -m venv .venv-analysis
.venv-analysis/bin/pip install -r requirements-analysis.txt
.venv-analysis/bin/python tools/rebuild.py
```

Generated tables are in `paper/tables`, figures in `paper/figures`, and derived
statistics alongside their generators in `analysis`. The checked-in
`paper/reference-tables` are the manuscript's reference outputs, not a second
dataset. `tools/verify.py` checks unique episode coverage and the integer counts
behind the manuscript. `tools/rebuild.py` checks rebuilt numerical table entries
against those reference tables.

### 2. Run the experiments with local model services

See [running experiments](docs/RUNNING.md) for installation, serving, and commands.
`experiments/run.py` prints a plan by default. Only `--execute` calls a model.
New runs go to `outputs/reruns`, never overwrite the published results, and refuse
existing output directories. A diagnostic `--limit` run is not a formal study.

| Study | Published matrix | Formal observations |
|---|---|---:|
| RQ1/RQ2 controlled | 6 configurations × 6 guidance/feedback conditions × 45 tasks × 14 topologies × 3 paired seeds | 68,040 |
| RQ3 workflows | 6 configurations × 4 concurrent conditions × 2 domains × 3 topologies × 3 cases × 3 repetitions | 1,296 |
| RQ4 defenses | 6 configurations × 4 defense arms × 45 tasks × 3 topologies × 3 paired seeds | 9,720 |

The six configurations are Qwen3-14B, Qwen3.5-9B without/with thinking,
Ministral-3-14B, GPT-OSS-20B (low reasoning effort), and Llama-3.1-8B.
`system`/`forward` in internal condition names mean task-level guidance
absent/present; both retain the fixed system-level handoff policy and canonical
task preservation. `G33-P3` is the internal graph ID for the paper's
`G⁺(3,3)`: `G(3,3)` with two additional target-reaching links. See
[study protocol](docs/PROTOCOL.md) for budgets and result denominators.

## Directory guide

- `experiments`: public launchers and published configuration matrices.
- `runtimes`: model-specific scientific code and necessary adapters. These copies
  are intentional: do not replace them with one model's runtime for every model.
- `results`: canonical controlled metrics, workflow traces, and final defense
  metrics/trajectories, split at complete JSONL records into files below 8 MB.
- `analysis`: only generators needed by the current paper and appendices.
- `provenance`: release checksums and original-to-release transformation hashes.
  Defense lineage records document why each canonical key was reused or replaced;
  they are not additional experimental observations.
- `tests`: artifact consistency tests; runtime-specific regressions are beside
  their source copies.

## Provenance and validation limits

Results are frozen observations, not generated examples. Operational path and
campaign-owner prefixes were anonymized in release copies; task payloads,
predictions, outcomes and numeric values were not selected or changed. Source
hashes distinguish the original files from the release copies. Canonical defense
results include reuse from unchanged pass paths and audited new executions, not
multiple observations for imported/retried records. No pilot/smoke observations
are included in the published denominators.

Later model runtimes retain their recorded frozen Git revisions. The original
Qwen3-14B local source mirror has no recoverable Git revision metadata; its release
files are SHA256-pinned. This is a provenance limit, not a claim of exact original
Git reconstruction. CPU/data checks do not establish bit-identical GPU inference
on a different machine. See `VALIDATION.md` for checks actually performed.

The release follows the manuscript's reporting cohort; it is not an exhaustive
survey of all exploratory models, nor a claim of preregistered cohort selection.
All historical materials are preserved separately by the authors.

## Licensing

Third-party packages and model artifacts retain their own licenses; no weights
are redistributed. See `THIRD_PARTY.md`. The authors have not yet selected a
license for their original code: see `LICENSE-STATUS.md` before public release.
