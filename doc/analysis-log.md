# Analysis log

Dated record of what was run, where, and what it showed. Newest entries at the top.

Add an entry whenever you run something whose result you would want to explain later —
it does not need to be long. Note the machine, the parameters, and the conclusion, and
link to the results directory if one was produced.

Template:

```
## YYYY-MM-DD — short title

**Ran:** what, with which parameters, on which machine
**Output:** path, if any
**Found:** the conclusion
**Next:** what it implies
```

---

## 2026-08-17 — Repository restructuring

**Ran:** No analysis. Declared dependencies in `pyproject.toml`, replaced all hard-coded
paths in `src/` and `scripts/` with `config.toml` profiles resolved by
`ambient_light_epilepsy.paths`, and added project documentation.

**Verified:** Reran `scripts/lux_analysis.py --limit 3` against the W: drive and compared
all metrics and covariates for the resulting 12 participants against the committed
`lux_5min_fmatch_analysis.csv` — identical to within 1e-9. The refactor changed path
handling only.

**Found:** Two latent bugs. `base_path` had meant the data root in `nhanes.py` and
`lux_metrics.py` but `data/{cycle}` in `cohort.load_pwe_seqn` and
`load_freq_matched_control_groups`, so notebook 09's calls pointed at the wrong location.
`find_people_on_asm` also wrote its output to `data/{cycle}/processed/` while every reader
looked in `data/processed/`. Both now resolve consistently.

Also confirmed that `PAXMIN_H.parquet` **does** exist on the W: drive — it is only absent
from the local partial copy — so the sparse H-cohort activity data noted in notebook 09
is not explained by a missing file.

**Next:** Phases 1 and 2 of [good-practice-plan.md](good-practice-plan.md).

---

## Reconstructed history

Entries below are reconstructed from git history, not written at the time. Dates are
commit dates and may lag the work. They are recorded because the project has already lost
one stretch of history to an undocumented gap.

**2026-05-13** — Commit "Unknown changes after not working on the project during April".
Content of these changes is not recoverable from the message; the project was paused
through April.

**2026-03-25** — `lux_metrics` extended to handle raw 1 Hz data as well as the 5-minute
downsample. Package functions changed to take a path parameter rather than hard-coding
one. The 1 Hz analysis was run on BlueBEAR, producing `lux_1hz_fmatch_analysis.csv`,
which notebook 08 reads. **No script in the repository generates this file.**

**2026-03-18** — Commit "Lots of edits". Results in `analysis/` date from around here.

**2026-03-11** — `time_above_threshold` changed to stop averaging across days, which had
been attenuating true time above threshold. `relative_amplitude` extended to return M10
and L5 midpoint times. Both changes alter reported metric values, so results produced
before this date are not comparable with results after it.

**2026-02-03** — Raw NHANES `.xpt` files converted to parquet on BlueBEAR. The commit
message records "No git trace of this" — the conversion step was not scripted in the
repository.

**2026-01-19** — Project started.
