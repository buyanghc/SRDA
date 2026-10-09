# Running the published experiments

All inference must run inside an appropriate allocated GPU job, not a shared
cluster login node. A single model service uses one GPU and an 8192-token context
window. The original controlled studies used L40 for Qwen3-14B and H200 for the
other configurations; the final defense campaign used L40. Hardware/serving
differences prevent a promise of bit-identical stochastic inference.

## Controller environment

Use Python 3.12 on Linux for the recorded controller stack. Create a separate
environment for each selected runtime, especially for model-family adapters:

```bash
python3.12 -m venv .venv-controller
.venv-controller/bin/pip install -e runtimes/qwen3.5-9B-nothinking
# Or select runtimes/<configuration>; RQ4 uses runtimes/defenses.
```

Recorded controller/service dependency locks are under each runtime's `nci/`
directory. Prefer those recorded Linux versions over unpinned upgrades. Do not
install CUDA service locks into a macOS plotting environment. Use a separate
vLLM environment for the model service. Model manifests identify recorded weight
revisions; no model files or author credentials are bundled.

## Model service

`experiments/serve.py` prints the selected service command. Supply the locally
downloaded model directory and the service interpreter/vLLM executable explicitly.
The Ministral profile uses the pinned vLLM 0.12.0 OCI image noted in its serving
specification; preserve its native Mistral tokenizer/load format and parser plugin.
GPT-OSS uses the frozen Harmony guard, including its exact parser-source check.
Llama uses the preserved parallel-call history template. Do not silently substitute
another precision, context size, parser or reasoning mode.

```bash
python3 experiments/serve.py --study controlled --model qwen3.5-9B-nothinking \
  --model-dir /your/models/Qwen3.5-9B --port 8010
# Inspect the command, then add --execute to start it within your GPU allocation.
```

Qwen3-14B's historical controlled-study weight revision was not recoverable from
the selected local metadata. Its RQ4 revision is pinned separately and must not
be presented as proof of the historical controlled-study revision. Do not claim
exact historical inference reproduction without resolving this provenance gap.

## Controlled RQ1/RQ2

Run each of six configurations and six conditions. One cell includes all three
paired graph/role seeds (1890 observations); `--seed` does not split this study.

```bash
.venv-controller/bin/python experiments/run.py controlled \
  --model qwen3.5-9B-nothinking --condition forward-natural
# Add --execute after the printed plan and model service are checked.
```

Conditions: `system-terminal`, `system-empty`, `system-natural`,
`forward-terminal`, `forward-empty`, `forward-natural`. The portable launcher
passes the exact frozen formal config to the original CLI. It permits an unpacked
source archive without Git metadata only after checking the release SHA256 files;
this bypass does not alter experiment settings.

## Concurrent RQ3 workflows

Four conditions per configuration, 54 observations each. Both domains and all
three topologies/cases/repetitions are included in one cell:

```bash
.venv-controller/bin/python experiments/run.py workflows \
  --model llama3.1-8b --condition forward-natural
```

The replication adapter passes each recorded model's thinking/reasoning settings.
The Qwen3-14B workflow runs through its original direct CLI. The legitimate and
attack requests share the environment but retain independent canonical tasks.

## RQ4 defenses

Install the `runtimes/defenses` controller. Start the selected model service with
`--study defenses` and the two pinned classifiers separately. Classifiers run on
CPU using the same tokenizer, label mapping and threshold as the recorded study.
Download classifiers at the exact revisions in `experiments/defenses/models.json`
into `<classifier-root>/<arm>/<revision>`. Use `tools/verify_classifiers.py` to
check all listed sizes and SHA256/Git-blob hashes and generate `VERIFIED.json`;
the classifier service refuses a mismatched receipt. Gated model access requires your own
upstream approval/token; never place the token in this repository.

```bash
python3 tools/verify_classifiers.py --models /your/classifiers --write
python3 experiments/defenses/classifier_server.py --models /your/classifiers --port 9010
.venv-controller/bin/python experiments/run.py defenses \
  --model llama3.1-8b --seed 0 --base-url http://127.0.0.1:8010/v1 \
  --classifier-url http://127.0.0.1:9010
```

Run all six configurations × seed indices 0,1,2. Each fresh cell executes all 540
planned keys; it does not require the authors' old minimal-rerun files. The runner
always selects final local-stop semantics. It preserves raw ERROR records and
diagnostic events, never retries until success, and stops on unclassified failures.
Completed coverage with retained errors is not a clean completion certificate.

## Tests and limits

`python3 -m unittest discover -s tests` needs no GPU. Runtime tests use pytest in
the installed controller environment (`pip install -r requirements-test.txt`).
Run `python -m pytest -q tests` inside the selected `runtimes/<configuration>`
directory; do not collect different runtime copies into one Python import context.
Artifact test scoping changes are documented in `provenance/artifact-adaptations.json`.
`--limit` is for diagnostics only and its
outputs must not replace published formal data. Fresh-run GPU smoke validation
has not been performed during local artifact preparation: no new inference was
authorized or used to manufacture the released results.
