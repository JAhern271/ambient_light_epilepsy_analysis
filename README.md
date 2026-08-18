# Ambient light exposure in epilepsy (NHANES)

Do people with epilepsy (PWE) experience different ambient light exposure than matched
controls, and does that help explain the circadian and sleep disruption reported in
epilepsy?

This repository analyses wrist-worn light sensor (LUX) and accelerometry recordings from
two NHANES cycles, comparing people identified as having epilepsy against
frequency-matched controls. See [doc/protocol.md](doc/protocol.md) for the full study
rationale, aims and planned methods.

**Status:** an exploratory light-exposure analysis is complete, based on the 1 Hz `PAXLUX`
recordings. Published results are instead intended to come from **`PAXMIN`**, which
carries minute-level light *and* activity for the same participants — sufficient
resolution for circadian-scale work, and it allows light and activity to be compared on
identical sampling. That analysis is in progress. See
[doc/data-sources.md](doc/data-sources.md) for how the two light sources differ,
[doc/good-practice-plan.md](doc/good-practice-plan.md) for outstanding work, and
[doc/analysis-log.md](doc/analysis-log.md) for the running record.

## Findings so far (exploratory, PAXLUX)

Comparing 192 PWE against 669 frequency-matched controls, adjusted for age, sex, PIR,
education, season and cohort, PWE show **lower daytime light exposure** (`mean_daytime_lux`),
**less time above 1000 lux**, **lower M10** and **lower interdaily stability**. Nighttime
light exposure does not differ between groups. Employment and depression each explain
part of the difference, but epilepsy remains significant after adjusting for both —
including for interdaily stability (p = 0.035), which an earlier, mis-specified IS
implementation had obscured. See [doc/analysis-log.md](doc/analysis-log.md).

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

Building the cohort is a script rather than a notebook, since everything downstream
depends on its output:

```bash
python scripts/build_cohort.py --dry-run
```

| Step | Purpose |
|---|---|
| `01 - Data exploration` | Convert raw NHANES `.xpt` files to parquet |
| `02 - Finding PWE` | Identify people with epilepsy from prescription data |
| `03 - Demographics of PWE` | Explore cases and check control balance (build via the script) |
| `04 - Timezone confirmation` | Verify LUX timestamps are local clock time |
| `05 - Cleaning LUX data` | Wear-time cleaning of PAXLUX (superseded by the PAXMIN route) |
| `06 - Testing metrics` | Develop and sanity-check the light metrics |
| `07 - LUX analysis` | Run the full PAXLUX analysis (executed on HPC) |
| `08 - LUX results` | Statistics and figures, PAXLUX — exploratory |
| `09 - PAX cleaning` | PAXMIN light and activity preprocessing — the live work |

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

- The PAXMIN-based analysis, which is the one intended for publication, is still in
  progress (notebook 09).
- **`PAXMIN_H.xpt` is an incomplete download and must be re-fetched from the CDC.** The
  file is full-size at 9.35 GB but only its first 2.99 GB carry data; the rest is zeros,
  so only 2,489 of 7,776 participants are present, and with them 40 of 110 cases and 130
  of 393 controls. Reconverting cannot help — the data is not in the file. Cycle G is
  complete. Run `python scripts/check_data_integrity.py` to check both the source `.xpt`
  and the converted parquet.
- PAXLUX-derived metrics are computed over the whole recording including non-wear time.
  The PAXMIN route does not share this problem, as it masks non-wear.
- The existing PAXLUX 5-minute files predate the parameterisation of the preprocessing
  scripts, so cycle G and H cannot be *shown* to have been processed identically. Low
  priority now that those results are exploratory.

## Licence

Not yet set — pending confirmation of institutional IP requirements.

## Citation

See [CITATION.cff](CITATION.cff).
