# Local artifact validation

Validated on 2026-10-09. These checks use frozen observations and CPU-only
deterministic/plan-generation tests; they are not a new GPU experiment.

## Passed

- Controlled results: 68,040 unique observations, 36 cells of 1890; identical
  paired keys, 14 topologies, all three seeds, five retained ERROR observations.
- Concurrent workflows: 1296 unique observations, 24 cells of 54; benign/attack/
  joint counts agree with the paper; all observations COMPLETED.
- Defenses: 9720 unique keys and 2430 complete four-arm planned pairings;
  9486 COMPLETED + 234 ERROR. Canonical raw outcomes agree with the metrics.
  Jointly completed comparison counts agree with the frozen summary. Lineage
  contains one provenance record per canonical key, not extra attempts.
- All 14 rebuilt numerical result tables match the current manuscript's
  reference numerical rows. Five active statistical figure PDFs rebuild.
- Five artifact test methods pass, covering the six configurations, fixed RQ4
  protocol, 18 launcher previews and 12 service-command previews.
- Six runtime suites: 74 passed each; defense suite: 39 passed, plus 17 subtests.
  Runtime-specific Python import contexts were isolated. These are deterministic
  CPU tests, not checks of model quality or fresh classifier inference.
- Actual native CLI plan preparation succeeds for all 78 formal cells:
  36 controlled, 24 workflow and 18 defense model/seed cells. No `--execute`
  inference switch was used; private plan-check outputs are outside this release.
- Release text was scanned for the source user's names, institutional paths,
  emails and common credential prefixes; none remain in the included files.
  Operational metadata was anonymized; synthetic benchmark identities remain.
- Every included file is smaller than 8 MB. Large JSONL files were split only
  between complete records. Verification checks each part's count and SHA256.

## Artifact-only adaptations

Launchers replace author-specific roots and cluster orchestration with explicit
local service/output arguments. The fresh RQ4 launcher calls the final frozen
scientific functions and local-stop implementation; it does not depend on the
authors' historical partial rerun directory structure. Source/result originals
remain untouched outside the release.

Original tests also covered unpublished single/fanout configurations and pilot
workflow files. Release tests scope those assertions to the six published
conditions and full workflows; they do not add those obsolete studies back into
the dataset. The defense deterministic test fixture dependency was included.
See `provenance/artifact-adaptations.json` for affected files and hashes.

## Limits and publication checks

- No fresh GPU/model-service/classifier smoke inference was performed during
  artifact assembly. Linux serving environments remain separate from the macOS
  CPU check environment. Cross-hardware bit-identical outcomes are not promised.
- Original Qwen3-14B controlled-study Git metadata and model weight revision
  were not recoverable from the selected local mirror. Runtime files are
  SHA256-pinned; the separate defense weight pin must not be relabeled as the
  historical controlled-study revision.
- Statistical values, not identical font/rendering bytes, were checked against
  the manuscript. Times New Roman is used when present; plot rendering may fall
  back to a system font elsewhere.
- The author-code license is still to be chosen. No license or public upload
  was made on behalf of the authors.
- Anonymity checks are local evidence, not a guarantee against every possible
  deanonymization channel. Review the selected GitHub repository metadata and
  the rendered anonymous mirror before submitting its link.

Use `python3 tools/verify.py` to recheck the frozen release. The explicit
`tools/freeze.py` utility updates the checksum inventory after deliberate edits;
verification and experiment launchers never regenerate checksums automatically.
