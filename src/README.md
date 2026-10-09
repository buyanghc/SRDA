# Frozen implementation

`gatepath/` is the shared reference implementation (final defense source profile).
Study launchers select the appropriate original profile, **not** this reference
blindly. Different recorded files reside under `variants/<profile>/`; identical
files are reused by their SHA256 digest. Tests, formal task data, configurations,
dependency locks and model adapters follow the same exact-byte deduplication.

`../configs/profiles/*.json` maps each original relative filename to its source
file and digest. `../tools/runtime.py` reconstructs and checks a profile before
the native CLI runs. Consequently model-specific differences are explicit while
seven copies of the same scientific code are not shipped.

The unchanged implementations contain helpers imported by the original package;
their presence does not add unreported experimental observations. The shared
reference profile is not a claim that every model used identical source bytes.
