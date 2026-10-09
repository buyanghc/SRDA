# Reviewer-edition validation

CPU-only checks performed during preparation (2026-10-09):

- All 438 original runtime assets reconstruct byte-for-byte from 98 shared files.
- Six model regression suites: 74 passed each; defense suite: 39 passed plus
  17 subtests. Release/launcher suite: six test methods passed.
- 78 original-CLI plan preparations passed: 36 controlled cells, 24 concurrent
  workflow cells, 18 defense cells. No model calls or inference results were made.
- Complete final matrices checked: 68,040 controlled, 1,296 workflow, 9,720
  defense observations, including all 234 defense ERROR records and 2,430
  matched four-arm cases. Canonical imported/reused keys are not double counted.
- All 14 numerical tables match the frozen manuscript reference rows; five
  statistical figures rebuild from released observations.

Compression changes storage only. Workflow records are explicitly a metric/
protocol projection, not full trajectories. The complete archived release remains
the source for trajectory inspection and original provenance. This edition does
not claim a new GPU smoke test, resolve historical model-revision gaps, or change
experimental settings.

Run `python3 review.py check` for release hashes and matrix/outcome checks, or
`python3 review.py figures` to reproduce the numerical table checks and plots.
