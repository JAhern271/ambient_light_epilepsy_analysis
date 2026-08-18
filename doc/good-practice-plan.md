# Plan: bringing this project into line with good scientific computing practice

Working plan for restructuring the repository against Noble (2009), *A Quick Guide to
Organizing Computational Biology Projects*, and Wilson et al. (2017), *Good Enough
Practices in Scientific Computing*. Tick items off as they land; this file doubles as
the project to-do list.

**Guiding test:** could someone else — or you in six months — clone this repository,
read one page, and regenerate every number in the paper?

When this plan was written the answer was no, for three reasons: the analysis inputs were
produced by code that was not in the repository, no result recorded which code version
made it, and nothing checked that the metric functions were correct. All three are now
addressed.

**Rescoped 2026-08-17.** PAXMIN carries minute-level light *and* activity for the same
participants, at a resolution ample for circadian-scale analysis, so published results are
intended to come from there rather than from the 1 Hz PAXLUX recordings. Work on the
PAXLUX pipeline is preserved and reproducible but is now exploratory. Phase 4b below
tracks the route that is actually headed for publication.

---

## Where the project already stands

Worth stating, because a fair amount is already right and should not be disturbed:

- Analysis logic lives in an installable package under `src/`, separate from notebooks.
- Raw and derived data are kept apart, and raw NHANES files are never edited in place.
- Notebooks are numbered, so the pipeline order is legible.
- `data/` is gitignored — no large or restricted data has been committed.
- Git history goes back to the start of the project.
- **Done 2026-08-17:** dependencies declared in `pyproject.toml`; all hard-coded paths
  removed from `src/` and `scripts/` in favour of `config.toml` + `paths.py`.

The gaps below are mostly about *provenance* and *verification*, not organisation.

---

## Phase 1 — Documentation (DONE 2026-08-17, except licence)

The single highest-value phase. Nothing here can break the analysis.

- [x] **`README.md`** — research question, key findings, install, path configuration,
      pipeline order, repository layout, known gaps.
- [x] **`doc/protocol.md`** — study rationale, aims and methods, with a
      *Deviations from protocol* table recording where the code differs from the design.
- [x] **`doc/data-sources.md`** — data dictionary: every NHANES table and variable used,
      and how each derived variable is defined.
- [x] **`doc/analysis-log.md`** — Noble's lab notebook, seeded with a reconstructed
      timeline from git history so the April gap is at least documented as a gap.
- [ ] **`LICENSE`** — still open, pending institutional IP confirmation.
- [x] **`CITATION.cff`** — drafted; ORCID, repository URL and licence field left as TODO.

## Phase 2 — Repository hygiene (DONE 2026-08-17)

- [x] **Removed the duplicated results file.** Confirmed the two copies were numerically
      identical (differing only in column order) before deleting one.
- [x] **Restructured results by date** into `results/2026-03-18/`, with a README recording
      what is known about their provenance and flagging what is not. Used `git mv` so
      history follows the files. The now-empty `analysis/` directory was removed and the
      `local` profile's `analysis_root` repointed to `results`.
- [x] **Driver script stamps its own output** — every run now writes a
      `.provenance.json` sidecar with git commit (plus `-dirty` flag), UTC timestamp,
      parameters, data root, machine, Python and package versions.
- [x] **Results now go to dated directories** automatically, so a run no longer silently
      overwrites the last one.
- [x] **`Finance simulation.ipynb`** — already removed in the `Before claude code` commit.
- [x] **Resolved the two stub notebooks.** Rather than deleting them and renumbering, both
      now carry a markdown cell stating their status: `05` records that LUX wear-time
      cleaning is unwritten and that metrics currently include non-wear time; `07` records
      that the analysis runs as a script and how to invoke it.
- [x] **Deleted the on-disk `.ipynb_checkpoints`** directories.

## Phase 3 — Testing (DONE 2026-08-17)

**This phase comes before any change to metric code**, including the outstanding protocol
items. A test written after a change can only confirm the new behaviour, not that it is
right.

- [x] **`tests/` set up with pytest**, added as a `dev` extra. 38 tests, running in
      under 7 seconds.
- [x] **Metrics tested against synthetic signals with known answers** — square wave,
      sinusoid, constant, alternating and white noise, each with hand-derived expected
      values for M10, L5, RA, midpoints, IS and IV.
- [x] **Synthetic example dataset** — built on the fly by the `example_data_root`
      fixture rather than committed as binary files, so the full loading path is
      exercised without the W: drive and nothing opaque enters git.
- [x] **Regression test** pinning real values for 6 cycle G participants, with
      `tests/regenerate_regression_fixture.py` to update it deliberately. Skips
      automatically where the real data is unreachable.
- [x] **Known-suspect areas checked.** See the 2026-08-17 entry in
      [analysis-log.md](analysis-log.md) — this turned up a real problem in
      `interdaily_stability`, described below.

### IS definition (RESOLVED 2026-08-17)

- [x] **Resolved by resampling to hourly before computing**, so numerator and denominator
      share a resolution, per Witting et al. (1990). Hourly is the convention in the
      nonparametric circadian literature and makes the 5-minute and 1 Hz analyses agree.
      A `bin_size` argument allows other resolutions.
- [x] **The finding survives and strengthens.** Recomputed for all 861 participants: IS
      remains lower in PWE, and now stays significant after adjusting for employment and
      depression (p = 0.035, previously p = 0.056). See [analysis-log.md](analysis-log.md).
- [x] **Regression fixture regenerated** deliberately, and tests now assert that IS is
      independent of input resolution.
- [x] **Confirmed no 1 Hz metric recompute is needed.** Corrected IS computed from 1 Hz
      recordings matches the 5-minute-derived value to 3e-06 (0.001%), because IS now
      resamples to hourly first.
- [x] **Notebook 08 regenerated** 2026-08-17. Its own models confirm IS remains significant
      after adjusting for employment and depression (p = 0.036, previously p = 0.079), and
      the written conclusion has been corrected. Also repointed from a hard-coded W: path
      to a `paths` lookup, and its stale "06" title fixed.
- [x] **Stale IS column replaced** in `results/2026-08-17/lux_1hz_fmatch_analysis.csv`,
      with a provenance sidecar recording that only IS was substituted and why.
- [ ] **Rerun `scripts/lux_analysis.py --downsample 1hz`** when convenient, so the 1 Hz
      results come from a single internally consistent run rather than a patched file.
- [ ] **Decide which resolution is reported.** `IV` legitimately differs between 5-minute
      and 1 Hz, so the two results files are not interchangeable wholesale.

## Phase 4 — Close the provenance gap

This was the most important phase scientifically: the analysis started from files nothing
in the repository knew how to make. Passes one and two closed that. Pass three has since
been rescoped around the move to PAXMIN.

### Pass one — recover and commit (DONE 2026-08-17)

- [x] **The preprocessing code was found on the W: drive**, not lost: three R scripts with
      Slurm submission scripts. Committed verbatim so the commit changes no behaviour.
- [x] **`.gitattributes` added** forcing LF on `.sh` and `.R`, so a script edited on
      Windows cannot reach BlueBEAR with CRLF and fail as `bad interpreter: /bin/bash^M`.
- [x] **Slurm job logs gitignored** (`*.out`, `*.err`, `*.stats`).
- [x] **`scripts/README.md`** documenting pipeline order, how to submit, and the known
      problems left unfixed.

### Pass two — parameterise (DONE 2026-08-17)

- [x] **Cohort is now an argument** to all three R scripts, with the submission scripts
      passing it through (`sbatch downsample_lux.sh G`). `convert_xpt` also takes an
      optional table list, and `downsample_lux` optional bin width and alignment.
- [x] **Absolute RDS paths removed.** `scripts/lib/ale_paths.R` is the R counterpart of
      `paths.py`: the `.sh` exports `ALE_PROJECT_ROOT` derived from its own location, and
      `ALE_DATA_ROOT` overrides the data directory. Both directory layouts are handled.
- [x] **`run_lux_analysis.sh` reconciled** — passes arguments through to the Python
      command-line interface, resolves the venv from `ALE_VENV` or `<root>/venv`, and
      fails with a clear message rather than silently using the wrong environment.
- [x] **Job logs include the job id**, so concurrent or repeated runs no longer overwrite
      each other's output.
- [x] **Safety improvements**: `convert_xpt` skips existing parquet unless
      `ALE_OVERWRITE=1`, every script prints its resolved paths before doing work, and
      omitting the cohort prints usage instead of running against the wrong data.

- [x] **Verified on BlueBEAR** 2026-08-17. `convert_xpt.R H PAXMIN` resolved both roots
      correctly on RDS, parsed arguments, and skipped the existing parquet. The data root
      resolves via the `<project root>/../data` candidate, since the repository is checked
      out beside the data there rather than above it.

Still unexercised: the Slurm submission path (the R was invoked directly), and
`convert_paxlux` / `downsample_lux` since parameterisation.

### Pass three — provenance of the raw inputs

Rescoped 2026-08-17: PAXMIN is now the intended basis for publication, so the PAXLUX
reprocessing items below dropped off the critical path. What remains matters *more*, since
PAXMIN arrives as an `.xpt` and its integrity is now load-bearing.

- [ ] **Write a manifest** of raw inputs with checksums, so a partial or corrupt download
      is detectable. Now the highest-value item here: it would settle whether `PAXMIN_H`
      genuinely has more missing data than `PAXMIN_G`, or was simply downloaded badly.
- [ ] **Script the NHANES download** — fetch the `.xpt` files from the CDC, recording URLs
      and checksums.
- [x] **Second clone on the W: drive reconciled** — its only uncommitted changes were
      whitespace, and it now tracks `good-practice-restructure`.
- [ ] ~~Rerun both cohorts through the parameterised scripts.~~ Deferred: only affects
      PAXLUX-derived results, which are now exploratory.
- [ ] ~~Decide on the binning alignment.~~ Deferred with the above; `center` versus `start`
      only applies to PAXLUX downsampling, and is now switchable by argument anyway.

## Phase 4b — the PAXMIN route

The analysis actually headed for publication. Listed here so the plan reflects where the
work is going, not where it has been.

- [ ] **Promote the PAXMIN preprocessing out of notebook 09** into `src/`, where it can be
      tested — non-wear detection, wear-block segmentation, and the masking of both
      activity and light.
- [ ] **Test it**, as was done for the light metrics. The same trap applies: a wrong wear
      threshold produces plausible numbers rather than an error.
- [x] **`PAXMIN_H` question resolved 2026-08-18: the file is truncated and zero-padded**, holding 2,489 of 7,776 participants. A conversion fault, not real missing data. Needs reconverting before any cycle H PAXMIN result. See [analysis-log.md](analysis-log.md).
- [ ] **Rerun the light metrics on PAXMIN light** (`PAXLXMM`). `lux_metrics.py` is
      source-agnostic, so this needs no new metric code — but note `IV` is resolution
      dependent, so minute-level values will not match the PAXLUX figures, while `IS`
      will, since it resamples to hourly.
- [ ] **Compare light against activity on identical sampling**, which is the reason for
      the move to PAXMIN and what the secondary aims require.

## Phase 5 — Notebooks (DONE 2026-08-17, except the output-versioning decision)

- [x] **All 23 hard-coded absolute paths replaced** with `paths` lookups, across notebooks
      01, 03, 04, 06 and 09 (08 was done earlier). Each substitution was verified to
      resolve to exactly the same file, so the notebooks were not re-executed and their
      stored outputs remain valid. The 8 remaining occurrences are inside stored *outputs*,
      not code, and will clear when those cells are next run.
- [x] **Provenance cell added** to the seven notebooks that run anything, calling
      `provenance.describe()` — commit, machine, Python and package versions, and the
      resolved roots.
- [x] **`src/ambient_light_epilepsy/provenance.py` added**, so notebooks and the driver
      script share one implementation instead of `lux_analysis.py` carrying its own
      `git_commit`.
- [x] **`notebooks/README.md` written** — run order, status of each notebook, the
      conventions, and which reruns need thought.
- [ ] **Decide how notebook outputs are versioned.** Stored outputs make notebooks
      valuable as a record but produce unreadable diffs. Options in Open questions.
- [x] **Frequency matching promoted out of notebook 03** into
      `ambient_light_epilepsy.matching`, driven by `scripts/build_cohort.py`, with 15
      tests. Verified to reproduce the existing cohort participant-for-participant in
      both cycles before anything else was changed. The PAXMIN preprocessing in `09` is
      the same job and is tracked under Phase 4b.

One behaviour change worth noting: notebook 03 previously wrote matched-control lists to
a copy inside the repository while the analysis read them from the W: drive. It now writes
to the canonical `<data root>/processed`, which is the location actually read. The
matching is seeded, so a rerun reproduces the same participants.

## Phase 6 — Environment reproducibility (1 hour)

- [ ] **Commit an `environment.yml`** capturing the conda environment, and a lock file
      with exact versions for the record. `pyproject.toml` now gives compatible ranges;
      a lock gives the exact set that produced the results.
- [ ] **Document the BlueBEAR setup** — modules loaded, how the env is created, how jobs
      are submitted — in `doc/`. Add the Slurm submission script if there is one.

---

## Deliberately out of scope

These are analysis decisions, not computing practice, and are tracked separately:
the outstanding protocol items (100 lux threshold, day–night contrast, 06:00–18:00
window, BMI and physical activity matching), the ASM-versus-self-report case definition
sensitivity check, mediation modelling, and the sleep metrics.

One dependency is worth respecting: **do Phase 3 before changing any metric code.**
Tests written after a change can only confirm the new behaviour, not that it is right.

## Order

**Phases are numbered in the order they should be done.** Work through them in sequence;
Phase 5 is the one exception and can be picked up piecemeal whenever a notebook is next
opened.

| Phase | | Status |
|---|---|---|
| 1 | Documentation | Done, except the licence |
| 2 | Repository hygiene | Done |
| 3 | Testing | Done — but raised an open question about the IS definition |
| 4 | Close the provenance gap | Passes one and two done; pass three rescoped to input checksums |
| 4b | The PAXMIN route | Not started — the analysis headed for publication |
| 5 | Notebooks | Done, except the output-versioning decision |
| 6 | Environment reproducibility | Not started |

## Open questions

1. **Licence.** MIT and BSD-3 are the usual choices for research code. Does Plymouth or
   the fellowship impose anything, and does the funder require a specific one?
2. **Do results belong in git?** The CSVs are small (under 1 MB) and having them
   versioned is genuinely useful. The alternative is Zenodo at submission. Keeping them
   is recommended, but under `results/YYYY-MM-DD/`.
3. **Notebook outputs.** Three options: keep as-is (good record, terrible diffs);
   `nbstripout` (clean diffs, loses the record); `jupytext` paired scripts (clean diffs
   and a readable `.py`, slight workflow overhead). Recommendation: jupytext.
4. **Standardise the two data layouts?** `paths.py` currently reads both the W:/HPC
   layout and the local copy. Converging on one would be cleaner, but means moving files
   on the W: drive — your call whether that is worth the disruption.
5. **Where is the LUX preprocessing code?** Does it exist on BlueBEAR or in a scratch
   file somewhere, or does it need rewriting from scratch? This determines whether
   Phase 4 is an afternoon or several days.
