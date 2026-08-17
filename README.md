# Ambient light exposure in epilepsy (NHANES)

Do people with epilepsy (PWE) experience different ambient light exposure than matched
controls, and does that help explain the circadian and sleep disruption reported in
epilepsy?

This repository analyses wrist-worn light sensor (LUX) and accelerometry recordings from
two NHANES cycles, comparing people identified as having epilepsy against
frequency-matched controls. See [doc/protocol.md](doc/protocol.md) for the full study
rationale, aims and planned methods.

**Status:** the primary light-exposure analysis is complete; rest–activity and sleep
metrics are in progress. See [doc/good-practice-plan.md](doc/good-practice-plan.md) for
outstanding work and [doc/analysis-log.md](doc/analysis-log.md) for the running record.

## Key findings so far

Comparing 192 PWE against 669 frequency-matched controls, adjusted for age, sex, PIR,
education, season and cohort, PWE show **lower daytime light exposure** (`mean_daytime_lux`),
**less time above 1000 lux**, **lower M10** and **lower interdaily stability**. Nighttime
light exposure does not differ between groups. Employment and depression each explain
part of the difference, but epilepsy remains significant for the light metrics; for
interdaily stability it does not survive adjustment for both.

## Data

| Cycle | Years | Code |
|---|---|---|
| NHANES G | 2011–2012 | `G` |
| NHANES H | 2013–2014 | `H` |

NHANES data is public domain (US CDC/NCHS) but is **not** committed to this repository.
See [doc/data-sources.md](doc/data-sources.md) for the tables and variables used, and how
every derived variable is defined.

Raw data lives outside the repository, in one of three locations depending on the
machine — see *Configuring paths* below.

## Installation

```bash
pip install -e ".[notebooks]"
```

Add the `convert` extra if you need to convert raw NHANES `.xpt` files to parquet, which
requires `pyreadstat`:

```bash
pip install -e ".[convert,notebooks]"
```

Developed against Python 3.10 (conda environment `ambient-light-epilepsy`).

## Configuring paths

No path is hard-coded. [config.toml](config.toml) defines a profile per machine — `hpc`,
`w_drive`, `local` — and the first profile that exists on disk is used, so the same code
runs unchanged everywhere.

To check what resolved on the current machine:

```bash
python -c "from ambient_light_epilepsy import paths; paths.describe()"
```

To override, set `ALE_PROFILE` to a profile name, or `ALE_DATA_ROOT` / `ALE_ANALYSIS_ROOT`
to explicit directories. Machine-specific settings that should not be committed go in
`config.local.toml`, which overrides `config.toml`.

Two directory layouts are supported, because the complete dataset on the W: drive and the
partial local copy are arranged differently:

```
W: drive / HPC                     local copy
data/G/DEMO_G.parquet              data/G/raw_parquet/DEMO_G.parquet
data/processed/*.csv               data/G/processed/*.csv
data/PAXLUX_G/parquet_5min/        (LUX recordings are not copied locally)
```

The local copy is sufficient for cohort definition work but not for the LUX analysis.

## Running the analysis

The notebooks are numbered in pipeline order. Steps 01–03 produce the cohort files that
everything downstream depends on; they only need re-running if the cohort definition
changes.

| Step | Purpose |
|---|---|
| `01 - Data exploration` | Convert raw NHANES `.xpt` files to parquet |
| `02 - Finding PWE` | Identify people with epilepsy from prescription data |
| `03 - Demographics of PWE` | Select frequency-matched controls |
| `04 - Timezone confirmation` | Verify LUX timestamps are local clock time |
| `05 - Cleaning LUX data` | Wear-time cleaning of LUX recordings (not yet written) |
| `06 - Testing metrics` | Develop and sanity-check the light metrics |
| `07 - LUX analysis` | Run the full analysis (executed on HPC) |
| `08 - LUX results` | Statistics and figures |
| `09 - PAX cleaning` | Physical activity preprocessing (in progress) |

The full analysis runs as a script rather than a notebook, since it processes several
hundred participants:

```bash
python scripts/lux_analysis.py --downsample 5min
```

Use `--limit N` for a quick smoke test; limited runs write to a separate filename so they
cannot overwrite real results. `python scripts/lux_analysis.py --help` lists all options.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The metric tests check values derived by hand from synthetic signals, so they verify
correctness rather than just pinning current behaviour. Tests needing the full LUX
dataset skip automatically where it is unreachable, so the suite passes on any machine.

`tests/data/regression_expected.csv` pins metric values for six real participants. If it
starts failing after an intentional change to a metric, regenerate it with
`python tests/regenerate_regression_fixture.py` and record why in the analysis log.

## Repository layout

```
config.toml     Per-machine data locations
doc/            Protocol, data dictionary, analysis log, plans
notebooks/      Numbered analysis notebooks
results/        Analysis outputs, in dated directories
scripts/        Driver scripts for long-running analyses
src/            The ambient_light_epilepsy package (all reusable logic)
tests/          Test suite
data/           Raw and derived data (gitignored, not distributed)
```

## Known gaps

- The LUX preprocessing that produces the 5-minute parquet files is not in this
  repository, so that step is not currently reproducible. This is the highest-priority
  outstanding item.
- There are no automated tests for the circadian metric functions.
- `PAXMIN_H` appears to have substantially more missing data than `PAXMIN_G`; whether
  this is real or a conversion problem is unresolved.

## Licence

Not yet set — pending confirmation of institutional IP requirements.

## Citation

See [CITATION.cff](CITATION.cff).
