# Plan: bringing this project into line with good scientific computing practice

Working plan for restructuring the repository against Noble (2009), *A Quick Guide to
Organizing Computational Biology Projects*, and Wilson et al. (2017), *Good Enough
Practices in Scientific Computing*. Tick items off as they land; this file doubles as
the project to-do list.

**Guiding test:** could someone else — or you in six months — clone this repository,
read one page, and regenerate every number in the paper?

Right now the answer is no, for three reasons: the analysis inputs are produced by
code that is not in the repository, no result records which code version made it,
and nothing checks that the metric functions are correct.

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
- [ ] **Rerun the 1 Hz analysis** and regenerate notebook 08, whose displayed outputs and
      IS conclusion are now superseded.

## Phase 4 — Close the provenance gap (1–2 days, medium risk)

**This is the most important phase scientifically.** The analysis currently starts from
files nothing in the repository knows how to make.

### Pass one — recover and commit (DONE 2026-08-17)

- [x] **The preprocessing code was found on the W: drive**, not lost: three R scripts with
      Slurm submission scripts. Committed verbatim so the commit changes no behaviour.
- [x] **`.gitattributes` added** forcing LF on `.sh` and `.R`, so a script edited on
      Windows cannot reach BlueBEAR with CRLF and fail as `bad interpreter: /bin/bash^M`.
- [x] **Slurm job logs gitignored** (`*.out`, `*.err`, `*.stats`).
- [x] **`scripts/README.md`** documenting pipeline order, how to submit, and the known
      problems left unfixed.

### Pass two — parameterise

- [ ] **Make the cohort an argument** in `convert_xpt.R`, `convert_paxlux.R` and
      `downsample_lux.R`. All three hard-code cycle H behind an "EDIT THIS" banner, so the
      G outputs were produced by a version of the file that no longer exists. Until this
      is fixed, G and H preprocessing cannot be shown to have been identical.
- [ ] **Remove the absolute RDS paths** from the `.R` and `.sh` files, deriving the
      project root from the script location as the Python code now does.
- [ ] **Reconcile `run_lux_analysis.sh`**, which activates a venv inside a second clone of
      this repository on the W: drive and predates the Python script's command-line
      interface.
- [ ] **Reconcile the second clone** on the W: drive: 6 commits behind, with uncommitted
      changes to `scripts/lux_analysis.py`.
- [ ] **Script the NHANES download** — fetch the `.xpt` files from the CDC, recording URLs
      and checksums. Still undocumented.
- [ ] **Write a manifest** of raw inputs with checksums, so corruption or a partial
      download is detectable. This would settle the open question about the H cohort's
      PAXMIN data.
- [ ] **Document the centre-aligned binning decision** in the methods, and check whether
      it should be `start` rather than `center` given the hour-boundary windows.

## Phase 5 — Notebooks (half a day, low risk)

Can be done piecemeal, whenever a notebook is next opened.

- [ ] **Replace the 31 hard-coded absolute paths** across 6 notebooks with `paths.py`
      calls. Concentrated in `01 - Data exploration` (18), then `06` (5), `09` (3),
      `03` (2), `08` (2), `04` (1). Mechanical, and it makes every notebook portable
      between your PC, the W: drive and BlueBEAR.
- [ ] **Add a provenance cell** at the top of each notebook printing `paths.describe()`
      and the current git commit, so a saved notebook records the state it ran against.
- [ ] **Write `notebooks/README.md`** giving the run order and what each notebook is for
      — especially useful given the recent renumbering.
- [ ] **Decide how notebook outputs are versioned.** Stored outputs make notebooks
      valuable as a record but produce unreadable diffs. Options in Open questions.
- [ ] **Promote settled logic out of notebooks.** The frequency-matching functions in
      `03` and the PAXMIN preprocessing in `09` are real analysis code living in cells.
      Once stable they belong in `src/`, where they can be tested and reused.

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
| 4 | Close the provenance gap | Pass one done — scripts recovered and committed; pass two parameterises them |
| 5 | Notebooks | Not started — piecemeal |
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
