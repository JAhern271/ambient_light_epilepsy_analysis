# Scripts

Long-running steps that are submitted as jobs on BlueBEAR rather than run in a notebook.
The three R steps prepare the data; the Python step runs the analysis.

Every step takes its cohort as an argument and resolves its own paths, so the same file
serves both cohorts and nothing needs editing between runs.

## Pipeline order

| Step | Script | Does |
|---|---|---|
| 1 | `convert_xpt/convert_xpt.R` | NHANES `.xpt` to parquet (`haven` + `arrow`) |
| 2 | `convert_paxlux/convert_paxlux.R` | Per-participant LUX CSVs to zstd parquet, 1 Hz |
| 3 | `downsample_lux/downsample_lux.R` | Bin 1 Hz LUX to fixed-width means |
| 4 | `lux_analysis.py` | Compute the light metrics and join covariates |

Steps 1–3 are only needed once per cohort. Step 4 is the analysis proper.

## Running on BlueBEAR

```bash
git pull

sbatch scripts/convert_xpt/convert_xpt.sh G            # all standard tables
sbatch scripts/convert_xpt/convert_xpt.sh H PAXMIN     # or just one
sbatch scripts/convert_paxlux/convert_paxlux.sh G
sbatch scripts/downsample_lux/downsample_lux.sh G      # 5 min, centre aligned
sbatch scripts/downsample_lux/downsample_lux.sh H 1 start   # 1 min, start aligned
sbatch scripts/run_lux_analysis.sh --downsample 1hz
```

Every script prints the paths it resolved before doing any work, so the job log records
where the data came from. Omitting the cohort prints a usage message rather than running
against the wrong data.

Job logs are named with the job id (`convert_xpt_12345.out`) so concurrent or repeated
runs do not overwrite each other, and are gitignored.

## Configuration

| Variable | Effect |
|---|---|
| `ALE_PROJECT_ROOT` | Repository root. Exported by the `.sh` scripts; derived from the script's own location otherwise. |
| `ALE_DATA_ROOT` | Data directory. Defaults to `<project root>/data`, then `<project root>/../data`. |
| `ALE_OVERWRITE` | Set to `1` to make `convert_xpt` replace existing parquet files. |
| `ALE_VENV` | Virtual environment for step 4. Defaults to `<project root>/venv`. |

`lib/ale_paths.R` is the R counterpart of `ambient_light_epilepsy/paths.py`. Both
understand the two directory layouts, so the scripts run against the W: drive and the
HPC without change.

Steps 2 and 3 skip participants whose output already exists, so a job that hits its
walltime can simply be resubmitted.

## Methodological detail worth knowing

`downsample_lux.R` defaults to `center` alignment, so a timestamp in the 5-minute files
marks the **centre** of its bin, not the start: a sample labelled 06:57:30 covers
06:55:00–07:00:00. This matters at the day and night boundaries used by the metrics
(daytime is hours 07:00–18:59). Pass `start` as the third argument to bin from the
interval start instead.

## Still outstanding

- **The existing 5-minute files predate this parameterisation.** They were produced by
  editing a hard-coded cohort into the script, and the cycle G version of that file was
  never saved, so G and H preprocessing still cannot be *shown* to have been identical.
  Rerunning both cohorts through the current scripts would settle it.
- **No manifest or checksums** for the raw inputs, so a partial download is undetectable.
  Relevant to the open question about `PAXMIN_H` having more missing data than `PAXMIN_G`.
- **The NHANES download itself is not scripted.**

## Verification status

The shell scripts are syntax checked and their argument handling tested. **The R changes
have not been executed** — R is not installed on the Windows workstation, so they were
verified by inspection only. Run one cohort of one step first and check the paths printed
in the job log before trusting a full rerun.
