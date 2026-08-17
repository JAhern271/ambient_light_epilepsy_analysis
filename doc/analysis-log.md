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

## 2026-08-17 — Notebook 08 regenerated against corrected IS

**Ran:** Built `results/2026-08-17/lux_1hz_fmatch_analysis.csv` by replacing only the IS
column of the 1 Hz results with the corrected values (justified by the resolution
independence verified earlier), then re-executed notebook 08 end to end and updated its
written conclusions to match the new output.

**Result: the notebook's own models confirm the finding, using its sqrt transform and
HC3 robust standard errors.**

| Model | Old IS | Corrected IS |
|---|---|---|
| Unadjusted MWU | p = 0.0053 | p = 0.0037 |
| FDR corrected | p = 0.0133 | p = 0.0092 |
| Adjusted (sqrt, HC3) | coef −0.0196, p = 0.0112 | coef −0.0215, p = 0.0071 |
| Baseline, depression subset | p = 0.0368 | p = 0.0119 |
| + employment | p = 0.0581 | p = 0.0259 |
| + depression | p = 0.0612 | p = 0.0214 |
| **+ both** | **p = 0.0791** | **p = 0.0361** |

IS is significant in every model, including the full one. The notebook's previous
conclusion — that epilepsy stops predicting IS once employment and depression are
adjusted for — is now corrected in the markdown.

**One result moved the other way.** In the time-outdoors models, which use only the
n = 623 participants with a reported `minutes_outdoors` (136 PWE, down from 192),
epilepsy is *not* a significant predictor of corrected IS even at baseline
(p = 0.105; with outdoors p = 0.179). Under the old IS these were p = 0.026 and p = 0.071.
Given the smaller and non-randomly missing subset, this most likely reflects loss of
power rather than absence of effect, and the notebook now says so rather than claiming
either direction.

**Unchanged:** every non-IS metric is bit-identical to the previous run, confirming the
change was isolated to IS.

**Also:** notebook title corrected from "06 - Initial LUX analysis" to "08 - LUX results",
left over from the renumbering, and the hard-coded W: path replaced with a `paths` lookup.

---

## 2026-08-17 — IS definition corrected; the finding survives and strengthens

**Ran:** Rewrote `interdaily_stability` to resample the recording to hourly bins before
computing, so the numerator and denominator sit at the same time resolution, per Witting
et al. (1990). Recomputed IS for all 861 participants from the 5-minute data and reran
the group comparison against both the old and new definitions.

**Result: the finding holds, and is stronger under the corrected definition.**

| Model | Old IS | Corrected IS |
|---|---|---|
| Unadjusted (Mann–Whitney) | p = 0.0018 | p = 0.0037 |
| Adjusted for age, sex, PIR, education, season, cohort | coef −0.0187, p = 0.0050 | coef −0.0232, p = 0.0071 |
| Additionally adjusted for employment and depression | coef −0.0142, **p = 0.056** | coef −0.0202, **p = 0.035** |

IS remains lower in PWE throughout. Group means move from 0.172 / 0.152
(controls / PWE) to 0.299 / 0.276, consistent with the old implementation having
suppressed IS.

**This changes a stated conclusion.** Notebook 08 records that "after adjusting for
employment and depression, epilepsy is no longer a statistically significant predictor of
IS". Under the corrected definition it *is* still significant (p = 0.035). The earlier
non-significance was an artefact of the mixed-resolution implementation, which added
participant-varying noise to the measure. The adjusted effect is −7.8% of the control mean
(previously −10.9%).

**Verified: no 1 Hz recompute is needed for IS.** Corrected IS was computed directly from
the 1 Hz recordings for 6 participants and compared against the value derived from their
5-minute files. They agree to a **maximum absolute difference of 3e-06 (0.001%)** — the
residual comes from the centre-aligned 5-minute binning shifting a few samples across
hour boundaries. Because corrected IS resamples to hourly before computing, the source
resolution no longer matters, which is exactly the property the fix was meant to restore.

The corrected IS values computed here from the 5-minute data are therefore valid for the
1 Hz analysis too, and the stale IS column in `lux_1hz_fmatch_analysis.csv` can be
replaced without rerunning the metric computation.

**Caveats.**

- Only the IS column is affected. Every other metric is untouched by this change.
- `IV` genuinely differs between 5-minute and 1 Hz, so the two results files are still not
  interchangeable wholesale, and which resolution is the reported one remains a decision.
- Notebook 08's displayed outputs and its IS conclusion are superseded. Regenerating it is
  a rerun of statistics over an existing CSV, not a recompute of the metrics.
- `IV` is unchanged and remains resolution dependent by nature, so 5-minute and 1 Hz IV
  values still cannot be compared.
- One participant yields NaN IS (no variance); n = 860 adjusted, 781 with employment and
  depression included.

**Also:** `tests/data/regression_expected.csv` was regenerated deliberately, because the
IS column moved by design. All 41 tests pass. Tests now assert that IS is independent of
input resolution, which is the property the old implementation lacked.

**Next:** regenerate notebook 08 against corrected IS. A full 1 Hz metric rerun is not
required, though rerunning it once through `scripts/lux_analysis.py --downsample 1hz`
would produce a provenance-stamped results file under the dated results scheme.

---

## 2026-08-17 — Preprocessing scripts recovered and committed

**Ran:** No analysis. Located the missing preprocessing code on the W: drive under
`scripts/` and committed it verbatim, normalising line endings to LF and adding a
`.gitattributes` so shell scripts cannot be committed with CRLF and fail on Linux.

**Found:** The pipeline is **R**, not Python — `convert_xpt.R`, `convert_paxlux.R` and
`downsample_lux.R`, each with a Slurm submission script loading `R/4.5.0` and
`arrow-R/17.0.0.1` on BlueBEAR. This closes the largest reproducibility gap: the 5-minute
downsampling that every reported metric depends on is now under version control.

Three things the recovered code revealed:

1. **The cohort is hard-coded** in all three scripts, behind an "EDIT THIS" banner. The
   cycle G outputs were made by editing these same files and that version was never
   saved, so **G and H preprocessing cannot be shown to have been identical**. This is
   the strongest argument for the parameterisation planned in the next pass.
2. **Binning is centre-aligned** (`TIME_ALIGN <- "center"`), so a 5-minute timestamp marks
   the middle of its bin: 06:57:30 covers 06:55–07:00. Undocumented until now, and it
   shifts samples relative to the 07:00 and 20:00 window boundaries — differently in the
   5-minute and 1 Hz analyses.
3. **`run_lux_analysis.sh` activates a venv inside a second clone** of this repository on
   the W: drive. That clone is 6 commits behind and carries uncommitted changes to
   `scripts/lux_analysis.py`. It needs reconciling before it is pulled.

**Not changed:** the scripts are committed exactly as they ran, so this commit alters no
behaviour. Fixes are listed in `scripts/README.md` for the next pass.

**Next:** parameterise cohort and paths, then reconcile the second clone.

---

## 2026-08-17 — Test suite added, and a resolution problem in IS

**Ran:** Built `tests/` (38 tests) covering the light metrics against synthetic signals
with analytically known answers, path resolution across both directory layouts, and an
end-to-end run over a synthetic cohort. Added a regression test pinning real values for
6 cycle G participants.

**Found — needs a decision.** `interdaily_stability` computes its numerator from **hourly**
bins but its denominator from the **raw epochs**, so the two halves of the ratio are at
different time resolutions. The denominator therefore includes within-hour variance that
the numerator cannot capture, which pushes IS down.

Measured on 10 real cycle G participants, IS at matched hourly resolution is on average
**2.2x higher** than the implemented value (mean 0.201 vs 0.102). The ratio is **not
constant** — it ranges from 1.09 to 4.90 across participants — so this is not a simple
rescaling that cancels in a group comparison.

Two consequences:

1. IS values are not comparable with published figures computed at a single resolution.
2. IS is not comparable between this project's own 5-minute and 1 Hz analyses. The 1 Hz
   denominator carries far more high-frequency variance, so its IS will be lower again.

This matters because **reduced IS in PWE is one of the four headline findings**. The
direction of the effect may well survive — the group difference could be robust to how IS
is defined — but that needs checking rather than assuming.

**Not changed.** The metric code is untouched. Deciding whether to resample to hourly
before computing IS, or to use time-of-day bins at the epoch resolution, is a
methodological choice, and the tests now exist to make the change safely.

**Also found:**

- `intradaily_variability` is inherently resolution dependent too, so 5-minute and 1 Hz
  IV values cannot be compared either. Standard behaviour, but worth stating explicitly.
- `get_sampling_interval_minutes` infers the epoch length from the **first two samples
  only**. A recording that begins with a gap reports the wrong sampling rate, and every
  metric scaled by it is then wrong. Pinned by a test.
- IV returns NaN, not 0, for a perfectly constant recording (0/0). Relevant if a sensor
  ever fails and returns a constant.
- M10 midpoint on a tied plateau resolves to the earliest maximal window. Matters for
  synthetic or heavily rounded data, rarely for real recordings.

**Next:** decide how IS should be defined, then Phase 4.

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
