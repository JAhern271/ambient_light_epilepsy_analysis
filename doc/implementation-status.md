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

- [x] **Case definition is drug-first and has no reason-code requirement.** Done
      2026-09-02. `cohort.find_cases(cycle, definition=)` is code-first, with four
      definitions — `primary`, `narrow`, `broad`, `narrow_nocode` — all reading their
      drug lists from `analysis_params.toml` via the new `params.py`. Ascertainment
      raises if a G40-coded drug is on neither the ASM nor the blanked list, and raises
      for a code-requiring definition on a cycle without reason-code columns.
      `find_people_on_asm` warns and delegates to `broad`, selecting the same
      participants as before in both cycles. **The spec's §4.1 primary yield was
      mislabelled** — 72/56/46 was the count *before* non-ASMs were blanked; the
      confirmed figures are 70/54/44. Corrected in the spec, logged, and pinned by a
      test. Two open consequences below: the cases are not yet wired into matching, and
      `analysis_params.toml` is still unread outside `[cohort]`.
- [x] **Clock times are stored as linear minutes from midnight.** Done 2026-10-01.
      `relative_amplitude` now returns the M10 and L5 **start** times that §6.5 and §8.2
      specify, not midpoints. `compute_lux_summary` emits them as `m10_start_clock_min`
      and `l5_start_clock_min`: minutes past midnight, documented as circular in the
      docstring, in data-sources.md and in §8.2, for R to read with `circular`. Both
      choices were the researcher's. Two silent-misalignment bugs fixed in the same
      function: the epoch is now the most common timestamp gap rather than the first, and
      the 24 h profile sits on an explicit clock grid, so a missing bin can no longer
      shift later times. Regression fixture regenerated: the other 10 columns are
      bit-identical, and the new starts equal the old midpoints minus half a window
      (mod 1440) for all six participants. Group comparison stays in R (§9). The values
      in `results/` are still midpoints and still superseded. See the 2026-10-01 entry
      in [analysis-log.md](analysis-log.md).
      - [ ] **M10/L5 tie-break across midnight.** A tie goes to the first window found
            scanning from 00:00, so a 0-lux stretch from 22:00 to 06:00 gives an L5 start
            of 00:00, not 22:00 (start of the stretch) or 23:30 (centred in it). Common
            on lux, where a dark room reads exactly 0; rare on activity. Options: first
            window in the tied stretch, centre of the stretch, or report timing as
            undefined when tied. Pinned as current behaviour by
            `test_l5_tie_across_midnight_resolves_to_the_first_window_after_midnight`.
            Researcher's decision.
      - [x] **A clock time masked on every day takes every window containing it out
            of contention.** Done 2026-10-01. A window's mean is now taken over its
            non-NaN minutes, and the window is eligible if at least 20/24 of them are
            present (`rest_activity.min_window_coverage`, §6.5). It is a required
            argument of `relative_amplitude`, with no default. The researcher chose this
            over keeping the old behaviour, NaN for the participant, or GGIR's zero-fill.
            Measured first: 0 of 4,085 `eligible_H_primary_d04h20` participants (0 of 37
            cases) have such a minute on valid days, so it is a safeguard. Regression
            fixture bit-identical on all 13 columns. See the 2026-10-01 entry in
            [analysis-log.md](analysis-log.md).
      - [ ] **`relative_amplitude` (and IS/IV) use every row, not valid days only.** The
            threshold metric restricts to valid days; the nonparametric metrics do not.
            The participant-level runner should pass only valid-day minutes, or the
            functions should take the `wear.summarise_days` table as
            `minutes_above_thresholds` does.
      - [ ] **`intradaily_variability` has no NaN handling.** It uses `np.mean` and
            `np.diff` on the raw values, so a single masked minute makes IV NaN on
            PAXMIN. The six regression-fixture PAXLUX recordings have no NaN, so the
            test suite cannot show it.
            Found 2026-10-01; needs a rule (e.g. differences only between adjacent
            present epochs) and a synthetic test before IV runs on PAXMIN.
- [x] **Night window disagrees with the spec.** Done 2026-10-01. `compute_lux_summary`
      reads `light.day_window` and `light.night_window`; the `7, 19` and `20, 5` literals
      and both function defaults (including the third value, 22:00–05:00) are gone, so
      the window is now a required argument. One shared `in_clock_window` helper handles
      the midnight wrap and rejects impossible windows. Regression fixture regenerated:
      only `mean_nighttime_lux` moved, the other 11 columns bit-identical. See the
      2026-10-01 entry in [analysis-log.md](analysis-log.md).
- [ ] **Mean daytime/nighttime lux are reported as primary metrics.** Spec §6.2 rules
      them out as primaries because the sensor top-codes at 2,500 lux; they are retained
      only as caveated secondaries.

### Missing

- [x] **The case definitions are now the cohort the analysis uses.** Done 2026-10-01.
      `matching.eligible_participants` takes `definition` as a **required** argument —
      one of `cohort.DEFINITIONS` or `matching.LEGACY_DEFINITION` — so nothing inherits a
      definition, and an unknown one raises. `scripts/build_cohort.py` requires
      `--definition` and `--validity`; the superseded combination used to be reachable by
      saying nothing at all.

      **The output is the eligible analytic sample, not a matched set.**
      `eligible_{cycle}_{definition}_{rule}.csv` holds one row per eligible participant
      with an `epilepsy` flag. §8.1 matches on the propensity score in R and §9 puts the
      statistical layer there, so Python says who is eligible and R decides who is
      compared with whom. Frequency matching is retired and now runs only under
      `--frequency-match`, which exists to reproduce the February files.

      Built for five cohorts, all at `d04h20`:

      | cycle | definition | eligible | cases | controls |
      |---|---|---|---|---|
      | H | **primary** | 4,085 | **37** | 4,048 |
      | H | narrow | 4,085 | 22 | 4,063 |
      | H | broad | 4,085 | 97 | 3,988 |
      | G | broad | 4,032 | 81 | 3,951 |
      | G | `narrow_nocode` | 4,032 | 32 | 4,000 |

      Case counts reproduce the 2026-09-04 figures through a different code path, and
      three structural checks pass: the eligible pool is identical across definitions
      within a cycle, narrow cases are a strict subset of primary (22 of 37), and the
      legacy path still reproduces the February cohort exactly (110/393 in H, 82/276 in
      G). `freq_match_*.csv` untouched.

      **A correction to the earlier log.** The 2026-09-03 and 2026-09-04 entries report a
      cycle-H control pool of 3,984. That figure excludes the union of *all* definitions'
      cases, which approximates §4.2's **sensitivity** control definition ("excluding
      controls taking any ASM for a non-G40 indication"). The **primary** control pool is
      **4,048**, since §4.2 applies no ASM exclusion under the primary definition.
- [ ] **`analysis_params.toml` is read for `[cohort]` and `[validity]` only.** `params.py`
      is the mechanism; `[cohort]` was wired up 2026-09-02 and `[validity]` on 2026-09-03
      by `wear.py`. `[light]` windows on 2026-10-01; the rest of `[light]`, and `[sleep]`,
      `[matching]`, `[survey]` and `[multiplicity]`, are still specification-only. The
      remaining contradicting literal is the 1,000 lux threshold below. Note `min_valid_days` and `min_wear_hours` are settled
      but deliberately *not* read as defaults — see the valid-day item below.
- [x] **Non-wear and valid-day handling.** Done 2026-09-03, together with the
      participant-level validity item below. New `wear.py` implements §5.1 and §5.2:
      quality flag (`PAXQFM > 0`, equivalently any `PAXFLGSM` letter — verified to agree
      on all 88,223,479 rows), `PAXPREDM` non-wear and negative `PAXMTSM` (the `PAXTSM`
      floor was deleted on the same day as redundant),
      light and activity masked jointly from one array, noon-to-noon days with the partial
      first and last dropped, and the D-days-at-H-hours rule. 58 tests written first,
      against synthetic recordings with hand-derived answers, and checked by injecting
      four deliberate faults — one of which the tests initially missed, so the joint-
      masking fixture was rewritten. Verified against real data two ways: every
      minute-level count matches the CDC codebook exactly, and the day geometry agrees
      with independent `PAXHD` arithmetic for 30 participants across the whole `PAXFTIME`
      range. §5.1 masks 13.9% of minutes in cycle H. See the 2026-09-03 entry in
      [analysis-log.md](analysis-log.md).
      - [x] **Valid-day thresholds settled 2026-09-03: D = 4, H = 20**, following
            [Xiao_2023]. Recorded in `analysis_params.toml` but deliberately still
            **required arguments with no defaults**, so every run states the rule it
            applied and the sensitivity runs read identically to the primary one. Do not
            "fix" that by adding defaults. Now that outcomes are being produced against
            these values, changing either breaches pre-specification.
      - [x] **Validity tables built for both cycles**, 2026-09-03.
            `valid_recordings_{H,G}.csv` with provenance sidecars, in `data/processed`.
            **The cohort change is the opposite of what this item assumed.** It said
            switching "recovers cases"; it recovers a few and removes far more, because
            the old rule could not see non-wear:

            | | cycle H | cycle G |
            |---|---|---|
            | assessed | 7,776 | 6,917 |
            | valid, superseded 9-day rule | 7,537 | 6,608 |
            | valid, §5.2 | **6,385** | **5,926** |
            | admitted by the change | +74 | +125 |
            | excluded by the change | **−1,226** | **−807** |

            Case-level, which is what constrains the study: **primary 44 → 37** in cycle
            H, narrow 26 → 22, broad 115 → 97; cycle G broad 87 → 81, `narrow_nocode`
            35 → 32. Control pool 4,601 → 3,984 (H) and 4,385 → 3,951 (G).
            §4.1 and §7 corrected against the measured 37 — the spec had predicted
            40–46 and was wrong in direction.
      - [x] **`PAXPREDM == 4` ("unknown") is kept.** Decided 2026-09-03. 2,946,459
            minutes, 3.3% of the table. These are minutes of valid data with an uncertain
            *label*, not absent data, and the quality-flag rule already removes those the
            QC review rejected. §5.1 was amended to state the decision and its reasoning
            explicitly rather than leave it implied; pinned by a test.
      - [x] **`validity.min_valid_seconds` deleted.** Done 2026-09-03. The parameter is
            gone from `analysis_params.toml`, `PAXTSM` is no longer read at all, and §5.1
            now states that it is not an exclusion. Provably behaviour-neutral: the rule
            removed zero minutes not already excluded, in **both** cycles. A test asserts
            a 3-second minute is retained and that the parameter is absent, so re-adding
            the rule fails the suite.
      - [x] **Su 2022 valid-day rule built** 2026-09-04 as `d03h16`, both cycles.
            Primary 37 → 40 cases in H, broad 97 → 104, control pool 3,984 → 4,237;
            cycle G broad 81 → 87. The two rules disagree on 5.7% of cycle H. Su is a
            strict superset of the primary rule — 0 participants valid under `d04h20`
            but not `d03h16`, which is the monotonicity check the looser thresholds
            require. **Su's 40 cases are not a reason to make it primary**; that would
            be a post-hoc substitution after seeing the count.
            Validity tables now name their rule (`wear.rule_label`), so a sensitivity
            rule cannot overwrite the primary one, and `save_validity` refuses to
            replace an existing table without `--overwrite`.
      - [ ] **Johnson 2023 valid-day rule — deferred 2026-09-03, not dropped.** It needs
            structure the others do not: consecutive days, a total-daily-activity floor of
            200, and exclusion of a participant with **any** invalid day. Its wear
            threshold is the same 20 h (their ">4 h missing"). Deferred on the
            researcher's instruction. **Note it is currently pre-specified** in §5.2 and
            §8.4 item 2, so if the intention is to drop it rather than postpone it, that
            is a deliberate spec amendment and belongs in the list at the foot of this
            file — removing a pre-specified sensitivity analysis silently is exactly what
            the pre-specification exists to prevent.
- [x] **Participant-level validity rule.** Done 2026-09-03, folded into the item above
      because §5.2 replaces this rule rather than amending it. `PAXLDAY == '9'` is gone
      from `matching.eligible_participants`, which now takes `valid_seqns` as a
      **required** argument — the decision needs the 88-million-row PAXMIN table and
      depends on two unsettled thresholds, so no caller may reach it by default.
      `scripts/build_cohort.py` requires `--validity spec|legacy`. The superseded rule
      survives as `wear.header_only_validity` so the committed cohort files and everything
      in `results/` stay reproducible. The cohort change is **not** the one-directional
      recovery this item assumed: it also excludes participants with nine days of
      recording but too little wear, which the header rule could not see. Both directions
      to be measured once D and H are set — see above.
- [x] **Light thresholds at 100, 250 and 1,000 lux.** Done 2026-10-01, and it was not
      only the call site: the old function counted all 24 h rather than the day window,
      had no notion of valid days, and divided by every row, so on PAXMIN a masked minute
      would have counted as "not above". New `lux_metrics.minutes_above_thresholds` takes
      the `wear.prepare_minutes` frame and the `wear.summarise_days` table, counts retained
      day-window minutes strictly above each `light.day_thresholds` value per **valid**
      day, and averages across valid days. The superseded function survives for the frozen
      PAXLUX route, now reading `light.primary_day_threshold` (same value; fixture
      unchanged). Nothing yet calls the new function on real data — that waits for the
      participant-level runner.
      - [x] **Masked minutes inside the day window: raw.** Decided by the researcher
            2026-10-01, before any light metric was computed on real data. A masked
            minute counts as not above any threshold, and the day's count is not scaled
            up. Recorded as `light.masked_minutes` and in §6.2; pinned by a test. The
            function keeps `masked_minutes` as a required argument, so callers read it
            from `[light]`; `rescale` remains implemented and tested but is not part of
            the specification.
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
- [x] **State the valid-day selection check in §10.** Done 2026-10-01: added to §10.3
      after the non-differential sentence, wording drafted by Claude and approved as
      written by the researcher. The 2026-09-03 analysis-log entry
      tested whether the valid-day rule excludes cases differentially — it costs 7 of 44
      primary cases, and ASM sedation or disability reducing tolerance for a wrist device
      would make the rule select on something plausibly caused by the exposure. It found
      no consistent evidence: cases excluded 19.6% [10.7, 33.2] against controls 15.6%
      [14.6, 16.6] in cycle H, with cycle G running the *opposite* way (RR 0.66).
      §10.3's "if this measurement error is non-differential by epilepsy status it biases
      toward the null" assumes precisely what that check tested, so the result belongs
      next to it — and it is worth stating whichever way it had come out, since a reader
      cannot tell a check that was never run from one that found nothing. Wording is the
      researcher's; the numbers are in the log entry.
- [ ] **Draw the DAG** for the supplementary material (spec §12 item 9). The
      depression-as-mediator assumption in particular is arguable and should be inspectable.
- [ ] **Pre-registration** (OSF). Tag the spec in git at the point it is frozen, so
      "what did we pre-register" is answerable without archaeology.
