# Provenance

- Standalone repository name: `v3-health-benchmark`
- Extraction date: `2026-05-15`
- Source project: `causal-agent-benchmark`
- Source branch: `V2-healthcare`
- Source commit: `7c08b6098c65045c7455fe8f44824bbb85e77dd4`

## Scope of extraction

This standalone repository was created from the `v3_health/` healthcare
benchmark subtree in the source project. Only the healthcare-specific Python
package, its tests, and publication-oriented reproducibility assets were
carried forward.

Excluded source-project material includes:

- `v1/`, `v2/`, `v3/`, and `v3_5_swarm/`
- root-level benchmark artifacts unrelated to the healthcare benchmark
- transient caches such as `__pycache__/` and `.DS_Store`
- bulky raw benchmark outputs that are regenerable from the extracted package

## Extraction notes

- Packaging was normalized to a `src/` layout while preserving the import path
  `v3_health`.
- Tests were moved to top-level `tests/`.
- Default output paths were changed from `v3_health/results/...` to repo-root
  `results/...`.
- The curated release snapshot under `artifacts/release/` is regenerated from
  the extracted standalone CLI rather than copied wholesale from the source
  tree.
