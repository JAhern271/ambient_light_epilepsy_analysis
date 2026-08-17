# Notebooks

Numbered in pipeline order. Steps 01–03 define the cohort and only need rerunning if the
case definition or matching changes; everything downstream depends on their output.

| # | Notebook | Purpose | Status |
|---|---|---|---|
| 01 | Data exploration | Convert raw NHANES `.xpt` to parquet, inspect each table | Done |
| 02 | Finding PWE | Identify people with epilepsy from prescription data | Done |
| 03 | Demographics of PWE | Select frequency-matched controls | Done |
| 04 | Timezone confirmation | Verify LUX timestamps are local clock time | Done |
| 05 | Cleaning LUX data | Wear-time cleaning of PAXLUX | **Superseded** — see below |
| 06 | Testing metrics | Develop and sanity-check the light metrics | Done |
| 07 | LUX analysis | Placeholder; the analysis runs as a script | Reference only |
| 08 | LUX results | Statistics and figures for the PAXLUX analysis | **Exploratory** |
| 09 | PAX cleaning | PAXMIN light and activity preprocessing | **Live work** |

## Which of these matter

Published results are intended to come from **PAXMIN** (notebook 09), not from the PAXLUX
recordings. PAXMIN carries minute-level light *and* activity for the same participants,
which is ample for circadian-scale analysis and lets the two be compared on identical
sampling. See [../doc/data-sources.md](../doc/data-sources.md).

So notebooks 04–08 are a complete but exploratory line of work, and 05 is superseded
rather than merely unwritten: PAXMIN light is already masked for non-wear, which is what
05 would have had to do for PAXLUX.

Notebooks 01–03 remain fully relevant — they define who is in the study, independent of
which light source is used.

## Conventions

**No absolute paths.** Every notebook resolves data locations through
`ambient_light_epilepsy.paths`, so they run unchanged on the local machine, the W: drive
and BlueBEAR:

```python
from ambient_light_epilepsy import paths

paths.raw_table("G", "DEMO")        # the DEMO table for cycle G
paths.lux_file(62218, "G")          # one participant's 5-minute LUX
paths.lux_dir("G", "5min")          # the directory of them
paths.processed_dir("G")            # derived cohort files
```

**Provenance first.** Each notebook opens with a cell calling `provenance.describe()`,
which prints the commit, machine, Python and package versions, and the resolved data
root. Rerun it before saving so the stored outputs carry the state that produced them.

**Analysis logic belongs in `src/`.** Notebooks are for exploration and figures. Anything
settled should move into the package where it can be tested — see the plan.

## Reruns worth thinking about first

- **01** converts `.xpt` files to parquet. Rerunning overwrites them.
- **03** writes the matched-control lists. It now saves to the canonical
  `<data root>/processed`, which on the W: drive is the location the analysis actually
  reads — previously it wrote to a copy inside the repository. The matching is seeded
  (`default_rng(42)`), so a rerun reproduces the same participants.
- **08** reads a results CSV and is cheap to rerun.

Notebooks 01, 03, 04, 06 and 09 had their paths converted without being re-executed, so
their stored outputs predate that edit. The substitutions were verified to resolve to
exactly the same files, so the outputs remain valid.
