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

- [ ] **The new case definitions are not yet the cohort the analysis uses.**
      `cohort.find_cases` exists and is tested, but `matching.eligible_participants`
      still defaults to the legacy `people_with_epilepsy_{cycle}.csv` — the `broad`
      definition — and `scripts/build_cohort.py` has no `--definition` flag. Deliberate:
      the switch changes the study population, so it is its own commit. Wiring is one
      argument (`definition=`, already accepted by `eligible_participants`); the work is
      in deciding what to do with the existing `freq_match_*` files rather than in the
      code. **Nothing downstream uses the primary definition until this is done.**
- [ ] **`analysis_params.toml` is read for `[cohort]` and `[validity]` only.** `params.py`
      is the mechanism; `[cohort]` was wired up 2026-09-02 and `[validity]` on 2026-09-03
      by `wear.py`. `[light]`, `[sleep]`, `[matching]`, `[survey]` and `[multiplicity]` are
      still specification-only, and the literals that contradict them are the night-window
      and threshold items above. Note the two provisional `[validity]` thresholds are
      deliberately *not* read as defaults — see the valid-day item below.
- [x] **Non-wear and valid-day handling.** Done 2026-09-03, together with the
      participant-level validity item below. New `wear.py` implements §5.1 and §5.2:
      quality flag (`PAXQFM > 0`, equivalently any `PAXFLGSM` letter — verified to agree
      on all 88,223,479 rows), `PAXPREDM` non-wear, negative `PAXMTSM`, `PAXTSM` floor,
      light and activity masked jointly from one array, noon-to-noon days with the partial
      first and last dropped, and the D-days-at-H-hours rule. 58 tests written first,
      against synthetic recordings with hand-derived answers, and checked by injecting
      four deliberate faults — one of which the tests initially missed, so the joint-
      masking fixture was rewritten. Verified against real data two ways: every
      minute-level count matches the CDC codebook exactly, and the day geometry agrees
      with independent `PAXHD` arithmetic for 30 participants across the whole `PAXFTIME`
      range. §5.1 masks 13.9% of minutes in cycle H. See the 2026-09-03 entry in
      [analysis-log.md](analysis-log.md).
      - [ ] **The valid-day thresholds are still NOT settled — this is the live item.**
            `min_valid_days = 4` and `min_wear_hours = 20` follow [Xiao_2023] but the
            researcher reserved the choice pending a literature review (2026-09-02).
            Both are marked provisional in `analysis_params.toml` and are **required
            arguments with no defaults**: `scripts/build_validity.py` will not run
            without them, and the tests are parameterised on H ∈ {16, 20, 22} and
            D ∈ {3, 4, 5} rather than pinning 4/20, so settling them changes no test.
            **Ask the researcher for D and H before building or filtering a cohort.**
      - [ ] **Nothing has been run against a real cohort yet**, for the reason above. Once
            D and H are set: run `scripts/build_validity.py`, then report the cohort
            change in **both** directions before anything downstream uses it. From the
            header alone, up to 92 participants with `PAXLDAY < 9` have ≥ 4 candidate days
            and would be newly admitted; an unknown number with all nine days but heavy
            non-wear will now be excluded, which the old rule could not detect.
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
      - [ ] **Su 2022 valid-day rule** as a sensitivity analysis (§5.2, §8.4 item 2).
            Free — it is D=3, H=16 through the same required arguments, so it needs a
            run rather than any code.
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
