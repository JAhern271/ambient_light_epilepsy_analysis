# Results — 2026-08-17

> **Exploratory.** PAXLUX-derived. Published results are intended to come from PAXMIN
> instead — see `doc/data-sources.md`.

## Files

| File | Contents |
|---|---|
| `lux_1hz_fmatch_analysis.csv` | Per-participant light metrics plus covariates, both cohorts, PWE and matched controls. 861 rows. |
| `lux_1hz_fmatch_analysis.provenance.json` | How the file was produced. |

## What this is

The 1 Hz results carrying the **corrected interdaily stability** definition. Every column
except `IS` is the original 1 Hz value from the earlier BlueBEAR run; `IS` was recomputed
after `interdaily_stability` was found to mix time resolutions.

Corrected IS is independent of the source resolution — 1 Hz and 5-minute agree to 3e-06,
verified on 6 participants — so the values computed from the 5-minute recordings were
substituted rather than reprocessing 861 1 Hz files. See the 2026-08-17 entries in
`doc/analysis-log.md`.

This is what notebook 08 reads.

## Caveats

- **A patched file, not a single run.** The columns come from two computations. Rerunning
  `python scripts/lux_analysis.py --downsample 1hz` would produce an internally consistent
  file, and is worth doing if these results are ever written up.
- `IV` is unchanged and remains 1 Hz. It is inherently resolution dependent, so it is not
  comparable with 5-minute or minute-level IV.
