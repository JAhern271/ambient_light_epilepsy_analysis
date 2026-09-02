# Implementation status

> **Status:** the single to-do list for this project. Records where the code stands
> against the specification in [methods.md](methods.md). The spec says what the study
> does; this file says what is built, what is not, and what is built *wrongly*.
>
> **Revised:** 2026-08-27. **Supersedes:**
> [archive/good-practice-plan.md](archive/good-practice-plan.md) (whose live items are
> carried over below).
>
> Rule: when code and spec disagree, fix the code or change the spec deliberately — never
> silently edit one to match the other. Log the decision in
> [analysis-log.md](analysis-log.md).

---

## The scope change of 2026-08-27

`RXQ_RX_G` (2011–2012) contains **no reason-for-use variables**. CDC never released them
for that cycle; only `RXQ_RX_H` carries `RXDRSC1–3`. The ICD-10 G40 requirement that the
spec's case definition depends on is therefore **implementable in cycle H only**. Both
prior NHANES epilepsy analyses (Tang 2024, Terman 2020) used 2013 onward, which is
consistent with this.

Measured yields, current-use prescriptions, after age ≥ 20 and a valid recording:

| Definition | Cycle | Identified | Age ≥ 20 | + valid recording |
|---|---|---|---|---|
| ASM name-list only (what the code does today) | G | 123 | 101 | 87 |
| ASM name-list only (what the code does today) | H | 157 | 136 | 115 |
| ASM name-list **+ G40** | H | 61 | 47 | 39 |
| **Any drug + G40**, ASM confirmed (spec primary) | H | 72 | 56 | 46 |

The name-list definition has a **PPV of 38.9%** against the G40 requirement in cycle H
(61/157). The 96 non-G40 cases are taking topiramate for migraine (`G43` ×26) and
divalproex/lamotrigine for mood disorders (`F31.9` ×23, `F39` ×19, `F32.9` ×15). The
name-list also *misses* genuine cases — lacosamide, clobazam and clonazepam/lorazepam/
diazepam all appear with G40 codes — which is why the code must become **code-first**
(select on G40, then confirm the drug is an ASM) rather than drug-first.

**Consequence:** the 192-case pooled cohort behind every result in `results/` is roughly
60% off-label. Those results are superseded, not merely exploratory.

**Decision taken:** cycle H only for the primary analysis, code-first G40 + ASM
confirmation. Cycle G is retained as a **labelled broad-definition replication cohort**
under the name-list definition. The G40-versus-name-list comparison in H is reported as an
empirical misclassification estimate for ASM-based epilepsy ascertainment in NHANES — a
gap the literature scan identified as unfilled.

---

## Blocking

*Nothing is blocked. The `PAXMIN_H` dependency cleared on 2026-09-02.*

- [x] **`PAXMIN_H` download.** Done 2026-09-02 via `scripts/fetch_nhanes.sh` with 16
      parallel connections, 16.7x the single-connection throughput.
- [x] **Reconverted and integrity-checked.** Both cohorts pass: 88,223,479 of 88,223,479
      records carrying data for H, 0 padding rows, 7,776 of 7,776 participants,
      **110/110 cases and 393/393 controls**, against 40 and 130 from the truncated file.
      Cycle G unchanged and complete. See the 2026-09-02 entry in
      [analysis-log.md](analysis-log.md).

## Spec ↔ code gaps

Ordered by how much damage they do if left.

### Wrong, not merely missing

- [ ] **Case definition is drug-first and has no reason-code requirement.**
      `cohort.find_people_on_asm` matches 12 ASM names. Rewrite as code-first with a
      `definition=` parameter (`primary` = G40 + ASM confirmation, cycle H;
      `broad` = name-list, for cycle G and for sensitivity; `narrow` = rarely-off-label
      ASMs) so all three of the spec's §4.1 sensitivity analyses come from one function.
      Cycle G can only ever serve `broad`.
- [ ] **Clock times are stored as linear minutes from midnight.** `m10_midpoint` and
      `l5_midpoint` in `lux_metrics.py`. Any group comparison of `l5_midpoint` reproduces
      the exact error the spec (§8.2) criticises in Tang 2024 and Bailey 2023, because L5
      straddles the wraparound. **The affected values are already in `results/`.** Fix the
      metric, then handle group comparison with circular statistics.
- [ ] **Night window disagrees with the spec.** Code uses 20:00–05:00
      (`lux_metrics.py:81`); the spec §6.1 fixes it at 23:00–06:00. The unused default on
      `compute_mean_nighttime_lux` is a third value (22:00–05:00). Resolve by moving both
      windows into `analysis_params.toml` and deleting the literals.
- [ ] **Mean daytime/nighttime lux are reported as primary metrics.** Spec §6.2 rules
      them out as primaries because the sensor top-codes at 2,500 lux; they are retained
      only as caveated secondaries.

### Missing

- [ ] **Non-wear and valid-day handling.** The PAXLUX route applies none at all — metrics
      span the whole recording, including non-wear. Notebook 09's PAXMIN masking
      (`PAXTSM < 45` or `PAXPREDM == 3`, wear blocks under 1,440 min discarded) is a
      different rule again, and neither applies a `PAXQFM` exclusion. Promote out of the
      notebook into `src/`, add the quality flag, noon-to-noon days, and the spec's
      ≥ 4 valid days at ≥ 20 h (§5.2). **Test before use** — a wrong wear threshold
      produces plausible numbers rather than an error.
- [ ] **Participant-level validity rule.** Code requires `PAXSTS == 1` *and*
      `PAXLDAY == '9'` (all nine days, including the partial first and last that the spec
      drops). This is stricter than the spec's rule, so switching recovers cases: 47
      age-eligible H cases become 39 under the current rule.
- [ ] **Light thresholds at 100 and 250 lux.** `time_above_threshold_normalized` already
      takes a `threshold` argument; only the call site is hard-coded to 1,000.
- [ ] **Proportion of daytime minutes at the 2,500 lux ceiling** (spec §6.2, secondary).
- [ ] **Day–night light contrast** (spec §6.2).
- [ ] **Categorical nighttime light** — none / low / high, split at the median among the
      exposed (spec §6.3), plus the hurdle-model sensitivity analysis.
- [ ] **Log-transform option for light IV**, with both raw and log reported (spec §6.4).
- [ ] **Rest–activity metrics on activity.** IS/IV/RA/M10/L5 currently run on lux only.
      The same functions run on `PAXMTSM`, so H4 is close to free once `PAXMIN_H` lands.
- [ ] **Survey weights.** `WTMEC2YR`, `SDMVSTRA` and `SDMVPSU` are not loaded anywhere.
      Required for the supplementary analysis (spec §8.5), halved for a 4-year sample —
      but note that with cycle H only, `WTMEC2YR` is used unhalved.
- [ ] **Propensity full matching** with balance diagnostics and effective sample size
      (spec §8.1). Keep `find_frequency_matched_controls` so existing results stay
      reproducible.
- [ ] **Matching is currently per-cycle.** With H as the primary cohort this mostly
      dissolves, but cycle must enter the propensity model for any pooled sensitivity
      analysis.
- [ ] **Pregnancy exclusion** (spec §4.3). No pregnancy variable is loaded.
- [ ] **Nested models 0–3, attenuation analysis, E-values** (spec §8.1, §8.3). Currently
      a single adjusted model in notebook 08.

### Known data-handling doubts

- [ ] **PHQ-9 sum may include refusal codes.** `nhanes.load_dpq` sums across all columns
      present; codes 7 (refused) and 9 (don't know) would inflate the total. Flagged in
      [data-sources.md](data-sources.md) and still unconfirmed.
- [ ] **`minutes_outdoors` (DEQ) is in the results file but in no version of the spec.**
      Decide its role — it is a plausible convergent-validity check on the light metrics
      and worth a line in §6, or it should be dropped.

## Carried over from the good-practice plan

- [ ] **`LICENSE`** — pending institutional IP confirmation. `CITATION.cff` has ORCID,
      repository URL and licence still as TODO.
- [ ] **Manifest of raw inputs with checksums.** Highest-value provenance item; it is what
      would have caught the `PAXMIN_H` truncation immediately.
- [ ] **`environment.yml` and a lock file.** `pyproject.toml` gives ranges, not the exact
      set that produced the results.
- [ ] **Document the BlueBEAR setup** — modules, environment creation, job submission.
- [ ] **Notebook output versioning.** Recommendation was jupytext; undecided.
- [ ] **Slurm submission path is unexercised** since parameterisation, as are
      `convert_paxlux` and `downsample_lux`.
- [ ] **Rerun `scripts/lux_analysis.py --downsample 1hz`** so the 1 Hz results come from
      one internally consistent run rather than a patched file. Low priority now that the
      PAXLUX route is exploratory and the cohort behind it is superseded.

## Retired

Kept working and reproducible, but off the path to publication.

- The **PAXLUX 1 Hz and 5-minute route**. Frozen. Still useful for the
  resolution-comparison appendix (IS is resolution-invariant, IV is not).
- **Frequency matching** (`matching.find_frequency_matched_controls`). Retained so the
  existing cohort files and results remain explicable.
- Everything in `results/`. Produced from the contaminated cohort, the old night window
  and the linear clock-time midpoints. Superseded on all three counts.

## Amendments still owed to the spec

Tracked here because they are edits to [methods.md](methods.md) rather than to code.

- [ ] **Sleep derivation contradicts itself** — §6.6 specifies PAXPREDM minute
      classification, §9 specifies GGIR with the van Hees algorithm. The literature scan
      is explicit that PAXPREDM should not serve as the sleep outcome without independent
      validation, so §9's choice is the supported one. Settle before anyone implements it.
- [ ] **`[LancetHL_2023]` author list** is unresolved (PMID 37148892).
- [ ] **`PAXLUX_G` documentation** was never checked against `PAXLUX_H`, particularly the
      2,500 lux ceiling that §6.2 depends on. Now lower priority — cycle G is a
      replication cohort only — but it still needs doing before the G results are reported.
- [ ] **Draw the DAG** for the supplementary material (spec §12 item 9). The
      depression-as-mediator assumption in particular is arguable and should be inspectable.
- [ ] **Pre-registration** (OSF). Tag the spec in git at the point it is frozen, so
      "what did we pre-register" is answerable without archaeology.
