# Re-running the paper experiments

Result checks and plot rebuilding need no model service. New inference requires
your own model access and an allocated GPU; never run it on a cluster login node.
The launchers print a plan by default. Only `--execute` starts inference, and
existing output directories are never silently overwritten or resumed.

## Controller and exact frozen profiles

```bash
python3 review.py runtime --profile qwen3.5-9B-nothinking
python3.12 -m venv .venv-controller
.venv-controller/bin/pip install -e .cache/runtimes/qwen3.5-9B-nothinking
```

Select one of the six keys in `experiments/models.json`; RQ4's controller uses
`--profile defenses`. Shared files in `src/` and each profile manifest reconstruct
the original frozen bytes. Materialized copies are ignored, checked before use,
and not the place to edit scientific code. Dependency locks and serving adapters
are in the selected profile's `nci/` directory. Use the recorded Linux controller
and separate vLLM service environment, not a macOS plotting environment.

## Model service

```bash
python3 review.py serve --study controlled --model qwen3.5-9B-nothinking \
  --model-dir /your/models/Qwen3.5-9B --port 8010
```

Inspect the printed command before adding `--execute` in a GPU allocation.
Supply `--vllm` or `--service-python` for your service environment. Preserve
8192-token context, precision, parsing adapters and reasoning/thinking mode.
Ministral uses the pinned vLLM 0.12.0 OCI image; GPT-OSS its Harmony guard;
Llama its parallel-call history template. Service commands retain these settings.
The historical studies used L40 for Qwen3 and H200 for other controlled profiles;
the final defense campaign used L40. Different hardware does not guarantee
bit-identical inference. Qwen3's historical controlled weight revision is missing
from recovered metadata; its pinned RQ4 revision is not evidence of that revision.

## Experiment plans

```bash
.venv-controller/bin/python review.py plan controlled \
  --model qwen3.5-9B-nothinking --condition forward-natural
.venv-controller/bin/python review.py plan workflows \
  --model llama3.1-8b --condition forward-natural
.venv-controller/bin/python review.py plan defenses \
  --model llama3.1-8b --seed 0 --classifier-url http://127.0.0.1:9010
```

Controlled cells include all three seeds (1890 executions); run all six conditions
per model: `system-terminal`, `system-empty`, `system-natural`, `forward-terminal`,
`forward-empty`, `forward-natural`. Workflow cells include all three repetitions
(54 executions); use the four conditions without `empty`. Defense cells include
540 executions; run all six models × `--seed 0`, `1`, `2`, with guidance and
Natural feedback fixed. Add `--execute` only after checking the plan and services.
`--limit` is a diagnostic subset, not a substitute for the formal matrix.

For RQ4, install the `defenses` profile and start both pinned CPU classifiers.
Download the revisions listed in `experiments/defenses/models.json` into
`<classifier-root>/<arm>/<revision>`, using your own approved access credentials.

```bash
python3 tools/verify_classifiers.py --models /your/classifiers --write
python3 experiments/defenses/classifier_server.py --models /your/classifiers --port 9010
```

The classifier service requires a verified receipt. Known capacity/invalid/
truncated detector results remain ERROR with their evidence; there is no
retry-until-success or guessed verdict. Unknown failures stop the fresh run.
The local-stop protocol and scoring definitions are in [PROTOCOL.md](PROTOCOL.md).

## CPU regression tests

Install `requirements-test.txt` in the selected controller environment, then run
`python -m pytest -q tests` from its `.cache/runtimes/<profile>/` directory.
Do not collect different model variants into one import context. These are CPU
tests; reviewer-edition preparation does not claim a new GPU inference run.
