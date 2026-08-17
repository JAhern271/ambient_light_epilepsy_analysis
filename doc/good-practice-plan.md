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

## Phase 3 — Notebooks (half a day, low risk)

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

## Phase 4 — Close the provenance gap (1–2 days, medium risk)

**This is the most important phase scientifically.** The analysis currently starts from
files nothing in the repository knows how to make.

- [ ] **Script the NHANES download.** `scripts/00_fetch_nhanes.py` — fetch the `.xpt`
      files for cohorts G and H from the CDC, recording URLs and checksums. Currently
      undocumented; the commit message notes the conversion was done on BlueBEAR with
      "no git trace of this".
- [ ] **Script the XPT to parquet conversion.** `nhanes.xpt_to_parquet` exists but no
      driver calls it over the full table list.
- [ ] **Script the LUX preprocessing — the biggest gap.** The analysis reads
      `PAXLUX_{G,H}/parquet_5min/SEQN_*.parquet`, roughly 7,000 files per cohort derived
      from the `*_Lux.tar.bz2` archives. No code in this repository produces them, so the
      5-minute downsampling — a decision that directly affects every reported metric — is
      currently unreproducible and undocumented. Recovering or rewriting this is the
      priority.
- [ ] **Write a manifest** of raw inputs with checksums, so corruption or a partial
      download is detectable. This would settle the open question about the H cohort's
      PAXMIN data.

## Phase 5 — Testing (1 day, low risk, high payoff)

Nothing currently verifies the circadian metrics. These are exactly the functions where
an off-by-one in a rolling window or a wrong denominator produces plausible numbers
rather than an error.

- [ ] **Set up `tests/` with pytest**, and add `pytest` as a dev extra.
- [ ] **Test the metrics against synthetic signals with known answers** — a perfect
      12h-on/12h-off square wave has analytically known M10, L5, RA and IS; a constant
      signal has IV = 0 and undefined RA; a single-day recording should behave sensibly.
- [ ] **Commit a small synthetic example dataset** so tests run without the W: drive.
      Wilson: "provide a simple example or test dataset".
- [ ] **Add a regression test** pinning current outputs for a handful of SEQNs, so
      refactoring cannot silently move results. The 12-participant comparison run during
      the path refactor is the template.
- [ ] **Check the known-suspect areas** while writing tests: `interdaily_stability` uses
      hourly bins against 5-minute samples in a way worth confirming against the
      Witting et al. definition; `intradaily_variability` assumes evenly spaced samples
      with no gaps, which fragmented wear may violate.

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

One dependency is worth respecting: **do Phase 5 before changing any metric code.**
Tests written after a change can only confirm the new behaviour, not that it is right.

## Suggested order

Phases 1 and 2 first — they are pure gain and cost half a day. Phase 5 next, because
tests protect everything after it. Then Phase 4, which is the real scientific gap and
the most effort. Phase 3 can be done piecemeal whenever a notebook is next opened.

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
