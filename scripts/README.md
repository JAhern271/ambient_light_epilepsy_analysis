# Scripts

Long-running steps that are submitted as jobs on BlueBEAR rather than run in a notebook.
The three R steps prepare the data; the Python step runs the analysis.

These were previously kept only on the W: drive and were not version controlled, so the
code that produced the analysis inputs could not be recovered. They are committed here
**verbatim** as a first step — the known problems listed below are deliberately left
unfixed so that this commit changes nothing about how the pipeline behaves.

## Pipeline order

| Step | Script | Does |
|---|---|---|
| 1 | `convert_xpt/convert_xpt.R` | NHANES `.xpt` to parquet (`haven` + `arrow`) |
| 2 | `convert_paxlux/convert_paxlux.R` | Per-participant LUX CSVs to zstd parquet, 1 Hz |
| 3 | `downsample_lux/downsample_lux.R` | Bin 1 Hz LUX to 5-minute means |
| 4 | `lux_analysis.py` | Compute the light metrics and join covariates |

Steps 1–3 are only needed once per cohort. Step 4 is the analysis proper and is the one
that gets rerun.

## Running on BlueBEAR

Each R step has a matching `.sh` Slurm submission script that loads the required modules
(`bear-apps/2024a`, `R/4.5.0`, `arrow-R/17.0.0.1`) and calls `Rscript`.

```bash
git pull
sbatch scripts/downsample_lux/downsample_lux.sh
```

Job logs (`*.out`, `*.err`, `*.stats`) are written next to the submission script and are
gitignored — one of them was over 1 MB.

Step 4 no longer needs a submission script edit to change what it does:

```bash
python scripts/lux_analysis.py --downsample 1hz
```

## Known problems, to fix in the next pass

Recorded here rather than fixed, so that this commit is a faithful snapshot of what
actually produced the current results.

1. **Cohort is hard-coded.** `downsample_lux.R` and `convert_paxlux.R` both point at
   `PAXLUX_H` behind an "EDIT THIS" banner, and `convert_xpt.R` at `PAXMIN_H.xpt`. The
   cycle G outputs were produced by editing these same files, and that version was never
   saved. **The G and H preprocessing cannot currently be shown to have been identical.**
   Should become a command-line argument.
2. **Absolute RDS paths** are baked into every `.R` and `.sh`. They should derive the
   project root from the script location, as the Python code now does via `config.toml`.
3. **`run_lux_analysis.sh` is superseded.** It activates a venv inside a *second clone* of
   this repository on the W: drive, and runs `python lux_analysis.py` with the `cd`
   commented out, so it only works when submitted from its own directory. The Python
   script it calls has since gained a proper command-line interface.
4. **No manifest or checksums** for the raw inputs, so a partial download is undetectable.
   Relevant to the open question about `PAXMIN_H` having more missing data than
   `PAXMIN_G`.

## Methodological detail worth knowing

`downsample_lux.R` bins with `TIME_ALIGN <- "center"`, so a timestamp in the 5-minute
files marks the **centre** of its bin, not the start: a sample labelled 06:57:30 covers
06:55:00–07:00:00. This matters at the day and night boundaries used by the metrics
(daytime is defined as hours 07:00–18:59), and is not mentioned anywhere in the Python
code. It also means the 5-minute and 1 Hz analyses assign samples to hours slightly
differently.
