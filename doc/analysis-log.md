# Analysis log

> **Status:** history, append-only. Records what was run and what it showed at the time.
> Entries are never edited to reflect later understanding — a superseding entry is added
> instead. Nothing here is a specification; see [methods.md](methods.md) for that.

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

## 2026-10-01 — M10/L5 clock times: starts, in circular minutes past midnight

**Ran:** `pytest tests` (195 pass, including the real-data regression test). This PC,
W: drive data, six cycle G PAXLUX participants. `tests/regenerate_regression_fixture.py`.
`results/` untouched.

**Decisions: the researcher's.**
1. **Start, not midpoint.** §6.5 and §8.2 name M10 and L5 start times; the code computed
   midpoints. Options as presented: because the window lengths are fixed, one is the other
   rotated by a constant (2 h 30 for L5, 5 h for M10). So circular means, dispersion and
   Watson–Williams results are identical either way. What differs: start matches nparACT
   (the §9 validation comparator) and Tang 2024's M10-start quartiles. M10 *midpoints*
   cluster around 13:00, exactly where a noon anchor would wrap. And midpoint would have
   required amending the spec. Start chosen, as the spec already says.
2. **Boundary representation: minutes past midnight, 0–1439, documented as circular.**
   Options presented were minutes past midnight, hours since noon (§8.2's alternative:
   linear-safe for L5 but not for M10 near noon), an angle in radians, and cos/sin
   components (the only representation whose arithmetic mean is correct). Columns are
   `m10_start_clock_min` and `l5_start_clock_min`. Nothing in the value prevents a plain
   mean; the column name, the docstring, data-sources.md and a new sentence in §8.2 carry
   the warning.
3. **Scope: fix the silent-misalignment bugs only.** The tie-break and masked-bin
   behaviours, and valid-day restriction, are new items in implementation-status.md.

**Problems found in `relative_amplitude` on minute data.**
- The epoch was read from the first two rows, so one missing minute at the start made a
  2-minute epoch and halved every window. *Fixed:* now the most common gap; an epoch that
  does not divide an hour raises.
- The 24 h profile was indexed by position among the clock times present, so a clock
  minute absent on every day moved every later time one epoch early. *Fixed:* explicit
  clock grid, reindexed so an absent bin is NaN in place.
- Ties go to the first window after 00:00, so a tied run across midnight starts at 00:00.
  *Open.*
- A clock time masked on every day makes every window containing it ineligible, without
  warning. *Open.*
- All rows are used, valid day or not. *Open.*
- Wrapping past midnight with a single clear minimum was already correct.

**Fixtures, agreed with the researcher before implementation.** 1-minute epochs, 7 days.
Clock fixture A: 1000 lux 08:00–18:00, 0 lux 22:30–03:30, 100 otherwise; by hand M10 = 1000
starting 480, L5 = 0 starting 1350, RA = 1. B (0 lux 22:00–06:00): L5 start 0, pinned as
current behaviour. C1 (one day masked 00:00–01:00): unchanged. C2 (12:00–12:59 masked
every day): M10 = 545 starting 780, by hand, pinned as current behaviour. D (second row
deleted) and D2 (05:00 absent every day): unchanged from A. E: L5 starts at 23:00 and 01:00
give 1380 and 60, arithmetic mean 720, circular mean 0.

**Equivalence.** Step 1: with only the epoch and grid fix, still returning midpoints, the
regression test passed on all 12 columns. Step 2: after the switch, HEAD's module and the
new one run in the same process give `==` equality on the other 10 columns for all six
participants. So does the old fixture, once read with `float_precision="round_trip"`; a
first comparison without that flag showed spurious mismatches from pandas' default CSV
float parser. The regenerated CSV's text is identical outside the two clock columns.

**Values that moved (second route: old midpoint − 300 or − 150, mod 1440, matched exactly).**

| SEQN | m10 midpoint → start | l5 midpoint → start |
|---|---|---|
| 62218 | 790 → 490 | 235 → 85 |
| 62282 | 760 → 460 | 150 → 0 |
| 62293 | 965 → 665 | 1420 → 1270 |
| 67368 | 760 → 460 | 150 → 0 |
| 65027 | 830 → 530 | 245 → 95 |
| 65217 | 805 → 505 | 280 → 130 |

62293's L5 now starts at 21:10 and the others' start between 00:00 and 02:10: the
wraparound §8.2 is about, inside a six-row fixture. Two participants start at exactly
00:00. Their L5 values are not exactly 0, so this is not shown to be the tie-break
artefact, but it is consistent with near-ties resolving that way.

**Deliberate faults**, each injected and reverted by script, source restored byte-identical:
start off by one (9 tests fail, including the regression test); window end reported as
start (11 fail); epoch from the first two rows (2 fail); profile without the clock-grid
reindex (1 fails, D2). The ±10-minute sinusoid tests do not catch the off-by-one; the
exact minute fixtures do.

**Next:** the tie-break and masked-bin decisions before rest–activity metrics run on
PAXMIN. The participant-level runner must pass valid-day minutes only.

---

## 2026-10-01 — Masked daytime minutes counted raw, not rescaled

**Ran:** `pytest tests`. No data run. `results/` untouched.

**Decision: the researcher's.** In the time-above-threshold metric (§6.2), a masked minute
inside the 07:00–19:00 window counts as not above any threshold, and the day's count is
not scaled up to the full window. Recorded as `light.masked_minutes = "raw"` and in §6.2.

**Options as presented.** On a day with 600 of 720 window minutes retained, 300 of them
above 1,000 lux: raw gives 300; rescaling gives 300 / 600 × 720 = 360. Raw means a day with
more masked daytime minutes can score lower for that reason alone. Rescaling assumes the
masked minutes resembled the retained ones, which fails if the device is removed at
characteristic times (bathing indoors, outdoor sport). The 2026-09-03 check found similar
retained fractions in cases and controls (median 0.96 and 0.95), measured over whole days
rather than the daytime window. A wear-only check of masked *daytime* minutes by group was
offered and not taken up.

**Pre-specification.** Taken before any light metric was computed on PAXMIN, so no
outcome had been seen. `rescale` stays in the code, tested, but is not part of the
specification and no sensitivity analysis was added.

## 2026-10-01 — Valid-day selection check stated in §10.3

**Ran:** nothing; spec edit only.

**What changed.** §10.3 now reports the 2026-09-03 differential-selection check next to
its "non-differential … biases toward the null" sentence, which assumes what the check
tested. Numbers are taken unchanged from that entry. The closing sentence — that 46 cases
can exclude only a large differential — is new interpretation, not in the original entry.

**Decision:** the researcher's. Claude drafted the wording; the researcher approved it as
written.

## 2026-10-01 — Daytime minutes above 100, 250 and 1,000 lux, per valid day

**Ran:** `pytest tests` (186 pass, real-data regression included). No data run. `results/`
untouched.

**What changed.** New `lux_metrics.minutes_above_thresholds` implements §6.2's primary
light metric: for each valid day, retained 07:00–18:59 minutes strictly above each of
`light.day_thresholds`, averaged across valid days. The superseded
`time_above_threshold_normalized` counted all 24 h, over the whole recording, with every
row in the denominator; it is kept, documented as superseded, for the frozen PAXLUX
route, and now reads its 1,000 lux from `light.primary_day_threshold` — the regression
fixture is unchanged, confirming the literal removal is behaviour-neutral.

**Left open, by the researcher.** How a masked minute inside the day window counts — not
at all (`raw`), or by rescaling the day's count to the full window (`rescale`). Both are
implemented behind a required `masked_minutes` argument with no default, so the metric
cannot be produced on real data until the choice is made.

**Tests.** Fixture A (two valid days plus an invalid one at 2,500 lux, and 1,500 lux at
02:00 outside the window): 360 / 240 / 180 minutes at 100 / 250 / 1,000, by hand.
Fixture B (120 of 720 window minutes masked): 300 raw, 360 rescaled. Second route through
`wear.prepare_minutes` and `wear.summarise_days` on synthetic PAXMIN: 120 minutes at every
threshold. Counting all 24 h instead of the window was injected as a fault and turned
Fixture A's 1,000 lux figure into 210, exactly as predicted; three tests failed.

## 2026-10-01 — Night window set to the specified 23:00–06:00

**Ran:** `pytest tests` (177 pass, including the cycle-G real-data regression test) and
`tests/regenerate_regression_fixture.py`. This PC, W: drive data. `results/` untouched.

**Output:** `tests/data/regression_expected.csv`, regenerated.

**What changed.** `lux_metrics.compute_lux_summary` now reads the day and night windows
from `analysis_params.toml [light]` (07:00–19:00 and 23:00–06:00, methods.md §6.1). It
used to pass 20:00–05:00 as a literal, and `compute_mean_nighttime_lux` carried a third,
unused default of 22:00–05:00. Both functions now require the window as an argument.

**Equivalence check.** Before regenerating, all 12 metric columns of the six pinned
participants were compared at rtol 1e-9: **only `mean_nighttime_lux` changed**; the other
11 were identical, and the regenerated file differs from the old one in that column alone.
New nighttime values are lower for five of six (e.g. 0.95 → 0.05 lux), consistent with
the 20:00–22:59 evening hours dropping out; one rises (1.82 → 2.04), so that participant's
23:00–05:59 is brighter on average than their 20:00–04:59. Not investigated further.

**Tests.** Hand-derived fixtures: night mean of hourly values 0–23 is 38/7; light only at
06, 19–22 h leaves both means at 0; a masked (NaN) hour is skipped, not counted as dark.
A second route through `wear.prepare_minutes` on synthetic PAXMIN gives exactly 100 lux
under the new window against 400 under the old one. Injecting an off-by-one into the
wrap-around (`<=` for `<`) fails four tests.

**Next:** the PAXLUX `results/` remain on the old window and stay superseded. Light
thresholds next.

## 2026-10-01 — Case definitions wired into the cohort; the analysis now uses the primary definition

**Ran:** `cohort.find_cases` for all five available cycle/definition combinations, then
`scripts/build_cohort.py` for each. This PC, W: drive data. 169 tests pass. `results/`
untouched; `freq_match_*.csv` untouched.

**Output:** `cases_{H,G}_{definition}.csv` (five, with provenance sidecars) and
`eligible_{cycle}_{definition}_d04h20.csv` (five, with sidecars), in `data/processed`.

**What changed.** `matching.eligible_participants` takes `definition` as a **required**
argument, one of `cohort.DEFINITIONS` or `matching.LEGACY_DEFINITION`; an unknown value
raises rather than falling through. `scripts/build_cohort.py` requires `--definition` and
`--validity`. Until now the superseded combination — the drug-first legacy case list and
the `PAXLDAY == '9'` validity rule — was reachable by saying nothing at all, which is how
a dry run reported 97 PWE when the specification's cohort is 37.

**The output is the eligible analytic sample, not a matched set.** Decided by the
researcher. §8.1 matches on the propensity score with `MatchIt` in R and §9 puts the whole
statistical layer there, so Python's job is to say who is eligible and which of them are
cases; R decides who is compared with whom. Frequency matching is retired and now runs
only under `--frequency-match`, which exists to reproduce the February files. Writing the
new cohort under names that carry the definition and the validity rule was preferred to
overwriting those files, because seven notebooks and scripts read them by name and would
otherwise have silently begun loading a different study population.

**Found.**

| cycle | definition | eligible | cases | controls |
|---|---|---|---|---|
| H | **primary** | 4,085 | **37** | 4,048 |
| H | narrow | 4,085 | 22 | 4,063 |
| H | broad | 4,085 | 97 | 3,988 |
| G | broad | 4,032 | 81 | 3,951 |
| G | `narrow_nocode` | 4,032 | 32 | 4,000 |

**Checks, all passed before the files were believed.**

1. **Second route to the case counts.** 37 / 22 / 97 / 81 / 32 reproduce the 2026-09-04
   figures exactly, through a different path: those were computed in memory by
   `find_cases(save=False)`, these by writing case files and reading them back through
   `load_cases`.
2. **Case-list counts match the pinned figures** — primary 70, narrow 38, broad 157 in H;
   broad 123, `narrow_nocode` 42 in G — the values in `data-sources.md` and in
   `tests/test_cohort.py`.
3. **The eligible pool is identical across definitions within a cycle** (4,085 in H,
   4,032 in G). It must be: eligibility is age and accelerometry validity, and has
   nothing to do with who counts as a case. A difference would have meant the case list
   was leaking into the pool.
4. **Narrow cases are a strict subset of primary** (22 of 37). Both require a G40 code and
   narrow restricts the drug list, so the nesting is forced.
5. **Equivalence of the legacy path**, per CLAUDE.md's rule about proving equivalence
   before changing behaviour: `--definition legacy --validity legacy --frequency-match`
   reproduces the February cohort exactly — **110 cases and 393 controls** in cycle H,
   **82 and 276** in cycle G, matching the committed files row for row.

**A correction to the two entries below.** They report a cycle-H control pool of
**3,984**. That number excludes the union of *all* definitions' cases, which approximates
§4.2's **sensitivity** control definition — "excluding control participants taking any ASM
for a non-G40 indication". The **primary** control pool is **4,048**, because §4.2 applies
no ASM exclusion under the primary definition: a participant taking topiramate for
migraine is an eligible control. Both numbers are correct for their own question; the
earlier entries called 3,984 the control pool without that qualification, and anything
quoting them should use 4,048 for the primary analysis.

**Next:** the §6 measures, none of which are built on PAXMIN yet — light thresholds at 100
and 250 lux, proportion at the 2,500 lux ceiling, day–night contrast, categorical
nighttime light, and the rest–activity metrics on `PAXMTSM`. Those produce the
participant-level CSV that §9 hands to R, which is the last thing standing between this
cohort and an analysis.

---

## 2026-09-04 — Su 2022 valid-day rule built as a sensitivity group; validity tables now name their rule

**Ran:** `scripts/build_validity.py --cohort H|G --min-valid-days 3 --min-wear-hours 16`,
the [Su_2022] rule pre-specified in §5.2 and §8.4 item 2. Both PAXMIN tables read end to
end again. This PC, W: drive data. 163 tests pass. `results/` untouched; `freq_match_*.csv`
not regenerated.

**Output:** `data/processed/valid_recordings_{H,G}_d03h16.csv` with provenance sidecars.
The primary tables were **renamed** to `valid_recordings_{H,G}_d04h20.csv` rather than
rebuilt — their contents and sidecars are unchanged and already record D=4, H=20, so a
second eight-minute pass would have produced identical files. Their sidecars therefore
predate the `rule_label` field the new ones carry.

**Filenames now name the rule.** `wear.rule_label(D, H)` derives `d04h20`, `d03h16`,
`d05h20` from the thresholds, so a sensitivity rule cannot overwrite the primary table and
a filename cannot disagree with the rule that produced it. Callers state which rule they
want — `wear.load_validity(cycle, label)`, and `scripts/build_cohort.py --validity spec`
now requires `--min-valid-days` and `--min-wear-hours`, which name the table it reads and
land in the cohort's provenance. `save_validity` also refuses to replace an existing table
without `--overwrite`, following `cohort._save_cases`: a validity table defines the study
population, so rewriting one silently would change what every downstream result was
computed from.

**Found.**

| | cycle H | cycle G |
|---|---|---|
| assessed | 7,776 | 6,917 |
| valid, 9-day rule | 7,537 | 6,608 |
| valid, **d04h20** (primary) | 6,385 | 5,926 |
| valid, d03h16 (Su) | 6,831 | 6,258 |
| disagreement between the two rules | 5.7% | 4.8% |

Case level, which is what constrains the study:

| definition | cycle | 9-day | **d04h20** | d03h16 | Su gain |
|---|---|---|---|---|---|
| **primary** | H | 44 | **37** | 40 | +3 |
| narrow | H | 26 | 22 | 25 | +3 |
| broad | H | 115 | 97 | 104 | +7 |
| control pool | H | 4,601 | 3,984 | 4,237 | +253 |
| broad | G | 87 | 81 | 87 | +6 |
| `narrow_nocode` | G | 35 | 32 | 35 | +3 |

**A logical check that passed, and would have caught a real bug.** Su is looser on *both*
thresholds (3 < 4 days, 16 < 20 hours), so every participant valid under the primary rule
must be valid under Su. Measured: **0 violations in both cycles** — `valid only under
d04h20` is zero, at population and at case level. Su is a strict superset. A single
violation would have meant the day-counting or threshold comparison was wrong; monotonicity
is one of the few properties of this rule that can be checked without knowing the right
answer, so it is worth asserting whenever a new rule is built.

Candidate-day distributions are identical between the two rules (7,522 at seven in H), as
they must be: candidate days are recording geometry and independent of D and H. A second
free consistency check.

**Named so it is not done quietly later.** Su yields **40** primary cases, back inside the
"approximately 40–46" that §4.1 originally claimed before measurement corrected it to 37.
That is **not** a reason to promote Su to primary. The thresholds were fixed before any of
this was measured, and substituting the looser rule *after* seeing that it gives a more
comfortable case count is precisely the post-hoc move pre-specification exists to prevent.
d04h20 remains primary at 37 cases; d03h16 is reported as the sensitivity analysis it was
always specified to be. Agreement between them is a stronger result than either alone.

**Also fixed, found by compile-checking rather than by the tests:** `build_validity.py` had
been left with an unterminated string literal — two newline escapes had been mangled into
real newlines by an earlier edit. The 155-test suite passed throughout, because nothing
imports the scripts. Worth remembering: `pytest` covers `src/`, not `scripts/`, so a script
change needs a compile or a run to be checked at all.

**Next:** Johnson 2023's rule remains deferred, and needs code the other two do not
(consecutive days, an activity floor, exclusion on any invalid day). The case-definition
wiring is still the substantive open item.

---

## 2026-09-03 — Is the valid-day rule a differential selection mechanism on cases? No consistent evidence

**Ran:** Descriptive comparison of wear and valid-day counts between cases and the
age-eligible control pool, from `valid_recordings_{H,G}.csv` plus a per-participant row
count over both PAXMIN tables. Cycle H under the `primary` and `broad` definitions and
cycle G under `broad`. Unweighted, unmatched, no formal inference — methods.md §9 puts
statistics in the R layer. This PC, W: drive data. Nothing written.

**Why it was asked.** §5.2 costs 7 of the 44 primary cases (see the entry below), and 37
is not simply a smaller 44 if the lost cases differ systematically. There is a plausible
mechanism: ASM sedation, disability, or anything reducing tolerance for a wrist device
would raise non-wear in cases, so the rule would be selecting on something plausibly
*caused by* the exposure. That also bears on §10.3, whose "non-differential measurement
error biases toward the null" reasoning assumes exactly what this checks.

**The comparison involves no outcome data** — only wear — so it carries no
pre-specification cost.

**Found: no consistent evidence of differential selection, and the direction flips.**

| | H primary | H broad | G broad |
|---|---|---|---|
| cases with accelerometry | 46 | 117 | 89 |
| controls | 4,794 | 4,723 | 4,575 |
| cases excluded by §5.2 | 19.6% [10.7, 33.2] | 17.1% [11.3, 24.9] | 9.0% [4.6, 16.7] |
| controls excluded | 15.6% [14.6, 16.6] | 15.6% [14.6, 16.6] | 13.6% [12.7, 14.7] |
| excess in cases | +4.0 pp (RR 1.26) | +1.5 pp (RR 1.10) | **−4.7 pp (RR 0.66)** |

Intervals are Wilson 95%, for reading the size of a difference rather than testing it. In
all three the case interval comfortably contains the control estimate, and cycle G runs
the opposite way — the signature of noise, not a mechanism.

Three further readings agree:

- **Median valid days is higher in cases**, not lower: 7.00 (IQR 5.25–7.00) against 6.00
  (5.00–7.00) in H primary.
- **Median retained fraction 0.96 in cases against 0.95 in controls.** If anything cases
  wore the device slightly better.
- **Geometry control passes.** Candidate days are 7.00 with IQR 7–7 in both groups, so
  recording length is identical between them and any difference in valid days is about
  wear rather than about how long the device ran. This is what separates the two.

**One pattern worth keeping, on numbers too small to lean on.** Cases under the primary
definition are more *polarised* than controls: 13.0% at zero valid days against 6.6%, but
52.2% at seven against 42.9%. That is the shape a small subgroup with more severe disease
or heavier sedation would produce. It rests on **6 cases** against roughly 3 expected, so
it is not evidence of anything; recorded so it can be looked at again if the sample ever
grows, and so it is not rediscovered as a surprise.

**The prior selection gate was checked too.** Device non-return removes 8 of 54
age-eligible primary cases (14.8%) and 19 of 136 broad cases (14.0%), against 904 of 5,627
controls (16.1%). Cases are marginally *more* likely to have accelerometry, so there is no
differential loss at that step either.

**A count stated precisely, because an earlier summary was loose:** 9 of the 46 primary
cases with accelerometry fail §5.2, of which 7 had been admitted by the superseded nine-day
rule. Both are consistent with 44 → 37; the other 2 failed the old rule as well.

**Next:** offered to the researcher as a candidate line in §10 — the check is worth stating
in the manuscript whichever way it had come out, and §10.3's non-differential assumption is
the natural place for it. Not added unilaterally.

---

## 2026-09-03 — Valid-day rule settled at D=4/H=20; it CUTS the cohort by 16%, the opposite of what was predicted

**Ran:** `scripts/build_validity.py --cohort H|G --min-valid-days 4 --min-wear-hours 20`,
dry run then for real. Reads both PAXMIN tables end to end (88,223,479 and 78,126,856
rows). This PC, W: drive data. First execution of `wear.valid_recordings` on real data.
155 tests pass. `results/` untouched; `freq_match_*.csv` deliberately not regenerated.

**Output:** `data/processed/valid_recordings_{H,G}.csv` with provenance sidecars.

**Decision taken by the researcher:** D = 4 valid days, H = 20 hours, following
[Xiao_2023] — the values carried as provisional since 2026-09-02. Both remain **required
arguments with no defaults** by explicit choice, so every run states the rule it applied
and the sensitivity runs read identically to the primary one.

**Found: the rule is far stricter than the spec assumed, and in the opposite direction.**

| | cycle H | cycle G |
|---|---|---|
| participants assessed | 7,776 | 6,917 |
| valid, superseded 9-day rule | 7,537 | 6,608 |
| **valid, §5.2** | **6,385** | **5,926** |
| admitted by the change | +74 | +125 |
| **excluded by the change** | **−1,226** | **−807** |
| net | −1,152 | −682 |

Case-level, which is what actually constrains this study:

| definition | cycle | 9-day rule | §5.2 | change |
|---|---|---|---|---|
| **primary** | H | 44 | **37** | **−7** |
| narrow | H | 26 | 22 | −4 |
| broad | H | 115 | 97 | −18 |
| broad | G | 87 | 81 | −6 |
| `narrow_nocode` | G | 35 | 32 | −3 |
| control pool | H | 4,601 | 3,984 | −617 |
| control pool | G | 4,385 | 3,951 | −434 |

**Why the spec's prediction failed.** §4.1 said the yield should be "approximately 40–46",
reasoning that §5.2 is *looser* than the nine-day rule because it drops the partial first
and last days. §5.2 is indeed looser about how many days must be **recorded** — it admits
74 participants who stopped early — but far stricter about how much of each day must be
**worn**. Non-wear is invisible to a header flag, so no reasoning from `PAXLDAY` could have
reached this; it had to be measured. §4.1 and §7 corrected; the yield table gains a
measured §5.2 column.

**Cross-checks. Every one passed before the numbers were believed.**

1. Participants assessed 7,776 (H) and 6,917 (G), and `header_only_valid` 7,537 and 6,608
   — all four match counts derived from `PAXHD` outside `wear.py`.
2. Candidate days per participant for H: 7,522 at seven, 147 below four — identical to the
   independent noon-boundary arithmetic done on 2026-09-03 before the module existed.
   Candidate days are pure geometry and independent of D and H, so this isolates the day
   construction from the threshold.
3. Stored `meets_criterion` recomputed from `n_valid_days >= 4`: 7,776/7,776 and
   6,917/6,917 agree.
4. **External.** 1,391 of 7,776 cycle-H participants fall below four valid days, 17.9%.
   [Xiao_2023] used this same rule on 2011–2014 and report n=7,013; against ~14,700 PAM
   participants of whom roughly 60% are adults, that implies ~20% attrition from validity
   plus missing covariates. Consistent, and it comes from outside this repository.
5. Spot-checked eight participants credited with 0 valid days by recomputing from PAXMIN:
   retained fractions of 4.3% to 60.2%, and no candidate day reaching 1,200 minutes. They
   are genuine non-wearers, not an artefact. (One of them is SEQN 73557, whose *geometry*
   the `late_start` test fixture borrows — the fixture copies its 11,529-minute shape and
   16:30 start, not its wear pattern.)

**One number that looks like a finding but is tautological:** all 1,226 (H) and 807 (G)
participants excluded by the change have `PAXLDAY == 9`. That is definitional — a
participant with `PAXLDAY < 9` was already excluded by the old rule, so cannot be
"excluded by the change". Recorded so it is not later read as evidence about wear.

**Pre-specification.** The thresholds were fixed before the yield was known and are **not**
revised now that it is. Whether the findings depend on them is what the pre-specified
valid-day sensitivity analyses are for (§5.2, §8.4 item 2): Su 2022 at D=3/H=16 needs no
new code, only a run.

**Next:** the case-definition wiring, which is the remaining half of "nothing downstream
uses the primary definition". `scripts/build_cohort.py --validity spec --cohort H` now runs
end to end, but with `definition=None` it still uses the legacy drug-first `broad` list and
reports 97 PWE rather than 37. Switching the validity rule and the case definition in one
step would make the cohort change uninterpretable, so that stays a separate commit.

---

## 2026-09-03 — `min_valid_seconds` deleted; `PAXPREDM == 4` retained; both decisions written into §5.1

**Ran:** Removed `validity.min_valid_seconds` from `analysis_params.toml` and the
corresponding rule from `wear.mask_minutes`; `PAXTSM` is no longer read at all and is out
of `MINUTE_COLUMNS`. Verified the deletion is behaviour-neutral by scanning **both** full
PAXMIN tables. This PC, W: drive data. 152 tests pass. Nothing in `results/` touched; no
cohort built.

**Output:** no data files. Changes to `analysis_params.toml`, `wear.py`, `test_wear.py`,
`methods.md` §5.1, `data-sources.md`, `implementation-status.md`.

**Found: the deletion changes nothing, in either cycle.**

| | Cycle H | Cycle G |
|---|---|---|
| Rows | 88,223,479 | 78,126,856 |
| `PAXTSM` minimum | 3 | 2 |
| `PAXTSM < 45` occurrences | 63 | 2,902 |
| Minutes it would **add** to the §5.1 exclusion set | **0** | **0** |
| `PAXQFM > 0` vs `PAXFLGSM != ''` disagreements | **0** | **0** |
| §5.1 masks | 12,302,429 (13.9%) | 8,962,404 (11.5%) |

The rule fires 46 times more often in cycle G than in cycle H and *still* excludes nothing
the three specified rules do not already exclude, so removing it cannot alter a result in
either the primary or the replication cohort. The `PAXQFM` / `PAXFLGSM` equivalence
established for cycle H also holds exactly in cycle G.

**Decisions taken by the researcher:**

1. **`validity.min_valid_seconds` deleted** rather than written into §5.1 as a fourth
   rule. It came from notebook 09 and was never specified. §5.1 now states positively that
   `PAXTSM` is not an exclusion, so the question does not reopen. A test asserts a
   3-second minute is retained and that the parameter is absent from `[validity]`, so
   re-adding the rule fails the suite.
2. **`PAXPREDM == 4` ("unknown") retained** — 2,946,459 minutes in H, 3.3%; 2,716,383 in
   G. These are minutes of valid data carrying an uncertain *label*, not minutes of absent
   data, and rule 1 already removes those the QC review rejected. Pinned by a test.
3. **Johnson 2023's valid-day rule deferred, not dropped.** It is still pre-specified in
   §5.2 and §8.4 item 2. Recorded as deferred in `implementation-status.md` with a note
   that dropping it outright would be a deliberate spec amendment, since removing a
   pre-specified sensitivity analysis quietly is what the pre-specification exists to
   prevent. Su 2022 needs no code — it is D=3, H=16 through the same arguments.

**Also:** §5.1 amended to *state* rather than change what it already specified — the
quality flag variable is named, and both decisions above are written down with their
reasoning. No rule changed. Recorded in the methods.md revision history.

**Next:** the valid-day thresholds D and H. The researcher has them; nothing has been run
against a real cohort until they are set.

---

## 2026-09-03 — Non-wear masking and valid-day rules implemented; PAXMIN read path verified against the CDC codebook

**Ran:** New `src/ambient_light_epilepsy/wear.py` implementing methods.md §5.1 and §5.2,
with `tests/test_wear.py` (58 tests) written before it and run against synthetic PAXMIN
recordings whose answers are derivable by hand. Read-only queries over all 88,223,479 rows
of `PAXMIN_H` and all 7,776 `PAXHD_H` records with `PAXSTS == 1`. This PC, W: drive data.
Nothing in `results/` touched; no cohort built or filtered.

**Output:** `src/ambient_light_epilepsy/wear.py`, `scripts/build_validity.py`,
`tests/test_wear.py`. No data files written.

**Found:**

1. **The read path is verified against the authority.** Every minute-level count computed
   from the parquet matches the frequency tables printed in the CDC codebook exactly:
   88,223,479 rows; `PAXPREDM` 47,112,094 / 26,073,307 / 12,091,619 / 2,946,459 for
   wake / sleep / non-wear / unknown; `PAXMTSM == -0.01` 1,875; `PAXFLGSM` non-blank
   274,027. This validates the XPT→parquet conversion and the column typing independently
   of anything in this repository.

2. **`PAXQFM > 0` and `PAXFLGSM != ''` are the same rule** — they disagree on **0 of
   88,223,479 rows**. `PAXQFM` is simply the number of letters in `PAXFLGSM`. §5.1's
   "contains any letter value" and CDC's own "values >0 indicate that this minute is
   invalid" are therefore interchangeable. Both are applied, as a union.

3. **§5.1 masks 12,302,429 minutes, 13.9% of the table**, almost all of it `PAXPREDM`
   non-wear. That is the quantity of data the PAXLUX route currently includes and should
   not, so it is the size of the problem this work fixes.

4. **`validity.min_valid_seconds = 45` is redundant, not merely rare.** `PAXTSM < 45`
   occurs 63 times in 88 million rows, and every one of those minutes is already excluded
   by rules 1 to 3, so the rule removes **zero** additional minutes from cycle H. It is
   retained but has no effect. (An earlier claim in conversation that `PAXTSM < 45` never
   occurs was wrong — the codebook gives the range as 3 to 60, and the minimum is 3.)

5. **Notebook 09's `minute_of_day` is wrong, and was never used.** It computes
   `PAXSSNMP % 1440`, which treats the first minute of the recording as midnight. PAXMIN
   carries no clock time; it must come from `PAXHD.PAXFTIME`, which ranges from 09:11 to
   21:30 in cycle H. Every participant's clock was therefore shifted by a different
   amount, up to 12.5 h. The column was never consumed downstream, so no result is
   affected, but it would have gone off the moment anyone applied the §6.1 day or night
   windows to PAXMIN. Reconstructing `PAXFTIME + PAXSSNMP` was checked against the day-1
   record count for 369 participants: **369/369 exact**.

6. **Noon-to-noon days: "drop the first and last" and "keep only wholly-covered days" are
   the same rule on this data** — 0 disagreements across all 7,776 participants, both
   giving 7 candidate days to 7,522 of them. The alternative of keeping every day and
   letting the 20 h threshold decide differs for 3,573 participants, 3,554 of whom have
   `PAXFTIME` ≤ 16:00: it makes the candidate-day count depend on whether the MEC
   appointment was before or after 4pm. Rejected for that reason, on the researcher's
   decision. Cost of the chosen rule: 147 participants have fewer than 4 candidate days
   versus 127 under the alternative, i.e. 20 more of 7,776 (0.26%).

7. **Two independent routes to the day geometry agree exactly.** `wear.py` working from
   real PAXMIN minute records, against arithmetic on `PAXHD` alone, for 30 participants
   spanning the whole `PAXFTIME` range and `PAXLDAY` 1–9: candidate-day count 30/30, full
   per-day coverage vector 30/30, coverage summing to the row count 30/30.

8. **A discrepancy chased and closed.** Reconstructing minute counts from `PAXHD` gave
   88,224,097 against the table's 88,223,479 — 618 too many, one minute each for 618
   participants. Cause: `PAXETLDY` is the *end* of the final minute, so a recording ending
   `12:39:00` has its last record at 12:38 while one ending `16:38:59` has its last at
   16:38. The fault was in the throwaway verification arithmetic, not in `wear.py`, which
   never reads `PAXETLDY`.

9. **A test hole found by mutation testing.** Four deliberate faults were injected to check
   the tests had teeth: an exclusive wear-threshold comparison (caught, 3 failures),
   dropping `PAXFTIME` (caught, 9), removing the quality-flag rule (caught, 7), and masking
   activity by non-wear alone rather than by the full retained set — **not caught**. The
   joint-masking fixture applied only non-wear, so the two masks were identical and the
   test could not distinguish them. Fixture rewritten to inject two different kinds of
   excluded minute; the mutation is now caught by 2 tests.

**Decisions taken by the researcher during this work:**

- Noon-to-noon days with the first and last dropped, per §5.2 as written (see 6 above).
- **`min_valid_days` and `min_wear_hours` are reserved pending a literature review.** The
  committed 4 and 20 are provisional. Consequence in code: both are required arguments
  with no defaults throughout `wear.py`, `scripts/build_validity.py` requires them on the
  command line, and the tests are parameterised on them (H ∈ {16, 20, 22}, D ∈ {3, 4, 5})
  rather than pinning 4 and 20, so settling them changes no test.

**Open question, not decided:** `PAXPREDM == 4` ("unknown", 2,946,459 minutes, 3.3%) is
kept, because §5.1 names only non-wear. Pinned by a test so that changing it is visible.

**Next:** `PAXLDAY == '9'` is gone from `matching.eligible_participants`, which now takes
`valid_seqns` as a required argument; `wear.header_only_validity` retains the superseded
rule so the existing cohort files reproduce. `scripts/build_cohort.py` requires
`--validity spec|legacy`. Nothing has been run against a real cohort: that needs D and H
first, and it will change the study population in **both** directions — from the header
alone, up to 92 participants with `PAXLDAY < 9` have ≥ 4 candidate days and would be
admitted, while participants with all nine days but heavy non-wear will now be excluded,
which the old rule could not detect. Both counts to be measured and reported before
anything downstream uses the new cohort.

---

## 2026-09-02 — Case ascertainment rewritten code-first; the spec's primary yield was mislabelled

**Ran:** Rewrote `cohort.find_people_on_asm` as `cohort.find_cases(cycle, definition=)`,
code-first per methods.md §4.1, with four definitions reading their parameters from
`analysis_params.toml` through a new `params.py`. This PC, W: drive data.

**Found: the spec's §4.1 primary row was the count before ASM confirmation.** It read
"Any drug + G40, ASM confirmed — 72 / 56 / 46". 72 is the number of participants carrying
a G40 code, *before* non-ASMs are blanked. Applying the confirmation step the spec
requires gives **70 / 54 / 44**. The other three rows of that table reproduce exactly
(61/47/39, 157/136/115 for H, 123/101/87 for G), so the error is specific to the
confirmation step. The spec table is corrected and now shows both rows.

Nineteen distinct drug names carry a G40 code in cycle H; sixteen are ASMs. The three
that are not cost two participants, because a non-ASM was their only G40 prescription:

- **SEQN 75016** — allopurinol coded G40, in a record otherwise of gout, type 2 diabetes
  and nerve pain. Their gabapentin is coded `M79.2`. A miscode.
- **SEQN 77740** — alprazolam, `RXDRSC1 = F41.9` (anxiety) with G40 secondary, alongside
  citalopram and amphetamine.
- **SEQN 78506** — hydrocodone coded G40, but this participant also has a confirmed ASM
  with G40, so only the row is blanked.

**Decisions taken by the researcher**, recorded as theirs: gabapentin with a G40 code
counts as a case (the code is the indication evidence, which is the basis of code-first
selection, even though gabapentin is excluded from the broad name list as non-specific);
alprazolam is blanked, while clonazepam, lorazepam and diazepam are retained, since those
three are used for seizure control and alprazolam is not; and the cycle G approximation
that §4.5 calls for gets its own name, `narrow_nocode`, rather than letting `narrow` mean
different things in different cycles.

**Design point worth keeping.** Confirmation by allow-list alone would silently discard
any G40-coded drug missing from the list — the same incompleteness that makes drug-first
selection wrong, displaced one step down the pipeline. So every drug observed with the
code must be on `asm_confirm` or on `non_asm_blanked`, and ascertainment **raises**,
naming the drug, if one is on neither. A new ASM or a new coding error forces a decision.

**Verification.** Equivalence before refactoring: `definition="broad"` selects exactly the
participants in the committed `people_with_epilepsy_G.csv` (123) and
`people_with_epilepsy_H.csv` (157), participant for participant. The identified counts
were reached twice by independent implementations — a throwaway pandas script and the
library — agreeing at 70 / 54 / 44 for `primary`, 38 / 31 / 26 for `narrow`. `tests/test_cohort.py`
adds 25 tests: a nine-row synthetic table whose answer under each definition is derivable
by hand, the completeness guard, refusal of a code-requiring definition on a cycle with no
reason-code columns, and a data-backed test pinning 70 / 38 / 157 that skips when the raw
data is unreachable. Full suite 90 passed.

**Also learned about the released data:** reason codes are three-character ICD-10
categories (plain `G40`, never `G40.909`) and an absent code is an empty string, not a
missing value. Every G40 row in cycle H is current use, so `RXDUSE == 1` changes nothing
for the code-first definitions, though it is still applied.

**Next:** nothing downstream uses the primary definition yet.
`matching.eligible_participants` accepts `definition=` but still defaults to the legacy
broad file, and `scripts/build_cohort.py` has no flag for it. That switch changes the
study population, so it is its own commit and its own decision about the existing
`freq_match_*` files.

---

## 2026-09-02 — PAXMIN_H downloaded, converted and verified complete

**Ran:** `scripts/check_data_integrity.py` against both cohorts, after the parallel
re-download and reconversion.

```
=== PAXMIN_G ===                          === PAXMIN_H ===
source .xpt : 78,126,856 / 78,126,856     source .xpt : 88,223,479 / 88,223,479
padding rows: 0                           padding rows: 0
participants: 6,917 of 6,917              participants: 7,776 of 7,776
SEQN range  : 62161-71916 (full)          SEQN range  : 73557-83731 (full)
cohort      : 82/82 cases, 276/276        cohort      : 110/110 cases, 393/393
```

**Found: cycle H is now complete.** All 110 cases and 393 controls are present, against
40 and 130 from the truncated file. The converted parquet is 912 MB against G's 871 MB,
the ratio expected given H has slightly more data; the broken version was 292 MB.

**What it took.** Four attempts, and each failure sharpened the tooling rather than just
costing time:

1. Two manual downloads produced full-length, zero-padded files. Both matched the
   advertised `Content-Length` exactly, because a transfer to a network drive
   preallocates from that header, so size checks could not detect either.
2. A reconversion with raised limits produced a byte-identical parquet, which showed the
   fault was in the source rather than the converter, and corrected an earlier diagnosis
   that had blamed the conversion.
3. `scripts/fetch_nhanes.sh` was written after measuring that the CDC throttles per
   connection rather than per client: sixteen connections gave 16.7x throughput, turning
   a 30-hour ETA into under two hours.
4. The first parallel run reached 89.6% before two connections timed out, which exposed
   that partial parts were discarded rather than resumed, and that curl's stderr was
   being sent to /dev/null so the failure was invisible.

**Guards now in place:** the source `.xpt` is checked for zero-filling before the parquet
is trusted, participant coverage is compared against PAXHD rather than relying on row
counts or file size, and the check reports which of source or conversion is at fault.

**Unblocks Phase 4b for cycle H.** With the spec now scoped to cycle H, this was the
gating dependency for the entire primary analysis.

---

## 2026-08-27 — Cycle G has no reason-for-use codes; study rescoped to cycle H

**Ran:** Compared candidate epilepsy case definitions in `RXQ_RX`, both cycles, on this
PC. Counted unique SEQN under each definition, then applied age ≥ 20 and the
accelerometry validity filters.

**Found: `RXQ_RX_G` contains no reason-for-use variables at all.** Cycle H carries
`RXDRSC1–3` and `RXDRSD1–3`; cycle G carries neither. Confirmed against the parquet on
both the local copy and the W: drive, and against the CDC codebook for RXQ_RX_G, which
documents seven variables and no reason-for-use fields. This is a property of the CDC
release, not of our conversion. Consistent with the literature: Tang 2024 used 2013–2014
and Terman 2020 used 2013–2016, both avoiding cycle G.

**The G40 requirement is therefore implementable in cycle H only.**

| Definition | Cycle | Identified | Age ≥ 20 | + valid recording |
|---|---|---|---|---|
| Any drug + G40, ASM confirmed | H | 72 | 56 | 46 |
| ASM name-list + G40 | H | 61 | 47 | 39 |
| ASM name-list only (current code) | H | 157 | 136 | 115 |
| ASM name-list only (current code) | G | 123 | 101 | 87 |

**The current case definition has a PPV of 38.9% against G40** (61 of 157 in cycle H). The
96 discordant cases are taking topiramate for migraine (`G43` ×26) and divalproex or
lamotrigine for mood disorders (`F31.9` ×23, `F39` ×19, `F32.9` ×15) — the off-label
pattern the literature predicts. The name list also *misses* genuine cases: lacosamide,
clobazam, and clonazepam/lorazepam/diazepam all appear with G40 codes, which is why
selection must be code-first rather than drug-first.

**This supersedes the existing results.** The 192-case pooled cohort behind everything in
`results/` is roughly 60% off-label ASM use. Those numbers are not exploratory-but-valid;
they are measuring the wrong group. The 123 and "22 of 123 under 20" figures in the old
protocol were cycle G alone, and 87 + 115 = 202 minus incomplete covariates gives the 192.

**Decision:** cycle H only for the primary analysis, code-first G40 + ASM confirmation.
Cycle G retained as a labelled broad-definition replication cohort, reported separately,
with attenuation toward the null expected and quantified by the measured PPV. If cycle G
shows an effect of *similar* magnitude to H, that is evidence against an
epilepsy-specific interpretation — pre-specified in methods.md §4.5 so that neither
outcome can be read as confirmatory after the fact.

**Next:** the PPV estimate is worth reporting in its own right — no NHANES-specific study
has quantified misclassification from ASM-based epilepsy ascertainment, a gap the
literature scan identified. Rewrite `cohort.find_people_on_asm` as code-first with a
`definition=` parameter. `PAXMIN_H` is now on the critical path for the whole primary
analysis, not just for cycle H.

---

## 2026-08-27 — The CDC throttles per connection; download scripted

**Ran:** A second manual re-download of `PAXMIN_H.xpt` failed worse than the first: 39,541
of 88,223,479 records carrying data, against 28,192,818 before. Both attempts produced a
file of exactly 9,351,691,760 bytes.

**Found:** that is precisely the `Content-Length` the CDC advertises. A transfer to a
network drive preallocates the full length from that header, so an interrupted download
leaves a complete-looking file with a zero tail. Two failures at different points, both
the "right" size.

Not a disk-space problem: 2.5 TB free on the share.

**The download is slow at source, not locally.** From an unrelated network the same URL
gives 91.7 KB/s, matching the 81 KB/s seen on BlueBEAR, and explaining the 30-hour ETA on
a single `wget`. Two simultaneous connections each sustained 92.5 KB/s, so **the throttle
is per connection, not per client**.

**Written:** `scripts/fetch_nhanes.sh` splits the file into byte ranges and fetches them
concurrently, roughly N times faster for N connections. Sixteen should bring 8.7 GB under
two hours. It assembles the output only once every part is present and the total matches
the advertised size.

Verified end to end against a 4 MB slice: the parallel result is byte-identical to a
single-connection fetch (same MD5), and took 17 s against 47 s. The resume path was
exercised by pre-seeding one complete and one truncated part — the complete part was
skipped, the truncated one refetched, and the result still matched.

**Note the URL is confirmed:** `https://ftp.cdc.gov/pub/NHANES/LargeDataFiles/PAXMIN_H.xpt`
returns HTTP 200, 9,351,691,760 bytes, `Last-Modified` 1 Aug 2022 — so the data has not
changed since cycle G was processed. No zipped version exists.

**Still outstanding:** the download itself, and everything in Phase 4b for cycle H.

---

## 2026-08-18 — Correction: PAXMIN_H is a bad download, not a bad conversion

**Supersedes the entry below.** That entry concluded the source `.xpt` was intact and the
conversion had truncated it. That was wrong, and the reconversion it recommended produced
a byte-for-byte identical file, which is what prompted a closer look.

**Ran:** Reconverted `PAXMIN_H` on BlueBEAR with the raised limits (128 GB, 4 h) and
`--overwrite`. Output was identical to the byte: 306,458,512 bytes, same 2,489
participants, same cut at SEQN 76872. Deterministic, so not a resource limit.

Parsed the `.xpt` headers directly and seeked into the file rather than trusting any
reader:

| | PAXMIN_G.xpt | PAXMIN_H.xpt |
|---|---|---|
| File size | 8,125,196,000 | 9,351,691,760 |
| Record length | 104 bytes | 106 bytes |
| Records the file spans | 78,126,856 | 88,223,479 |
| Records actually carrying data | 78,126,856 | **28,192,818 (32%)** |
| Last record's SEQN | 71,916 | **0** |
| Final 1,000 records | 72% zero bytes (normal) | **100% zero bytes** |

**Found:** `PAXMIN_H.xpt` is zero-filled from byte 2,988,441,668 onward. The file is the
right length but two thirds of it is empty. That is the signature of a transfer that
preallocated its final size and then stopped. R read it perfectly; there was nothing else
to read.

The row count reported by any reader is derived from file *length*, not content, so the
file presents as a valid XPT of 88.2 million rows. Nothing short of inspecting the bytes
would reveal it.

**Fix:** re-download `PAXMIN_H.xpt` from the CDC. No conversion change is needed; the
converter has been correct throughout.

**Guard added:** `integrity.check_xpt` parses the XPT header for the record layout, then
checks whether the tail is zero-filled and binary-searches for the last record carrying
data. `scripts/check_data_integrity.py` now checks the source before the parquet, and says
which of the two is at fault, since the parquet symptoms are identical either way.

**Also noted:** verifying against cycle G matters here. G's final records are 72% zero
*bytes* — normal, since much of the data is genuinely zero — while H's are 100%. Without
G as a control, "lots of zeros" would have been an ambiguous signal.

---

## 2026-08-18 — PAXMIN_H is truncated: a conversion fault, not missing data

**Ran:** Investigated the long-standing question of whether cycle H genuinely has more
missing accelerometry than cycle G. It does not. `PAXMIN_H.parquet` is **truncated and
zero-padded**.

| | PAXMIN_G | PAXMIN_H |
|---|---|---|
| Source `.xpt` | 7.6 GB | **8.7 GB** |
| Converted parquet | 913 MB | **306 MB** |
| Rows | 78,126,856 | 88,223,479 |
| Padding rows (SEQN = 0) | 0 | **60,030,661 (68%)** |
| Participants present | 6,917 of 6,917 | **2,489 of 7,776** |
| SEQN range | 62161–71916 (complete) | 73557–**76872**, should reach 83731 |

The source `.xpt` is *larger* than G's, so the data was downloaded. The parquet holds the
first 2,489 participants and then ~60 million rows of `SEQN = 0` with every value zero.
Row groups 0–26 are ~11 MB each; groups 27–84 are 0 MB.

**Impact on the cohort:** only **40 of 110 cases** and **130 of 393 controls** for cycle H
are present. Any cycle H analysis built on PAXMIN today silently uses a third of the
intended sample, biased toward low SEQN.

**Why nothing caught it.** The file opens cleanly, has no nulls, and has a *higher* row
count than G — padding makes a truncated file look bigger, not smaller. Every cheap sanity
check passes. Only participant coverage reveals it.

**Fix:** reconvert from the `.xpt`, which the parameterised script now supports:

```
ALE_OVERWRITE=1 sbatch scripts/convert_xpt/convert_xpt.sh H PAXMIN
```

Worth raising the job's memory and walltime first: at 8.7 GB this is the largest table in
the project, and a silent truncation is consistent with the conversion being cut short.
Rerun `scripts/check_data_integrity.py` afterwards to confirm.

**Guarded against recurrence:** added `ambient_light_epilepsy.integrity` and
`scripts/check_data_integrity.py`, which compare the participants actually present against
those PAXHD says should be there, and exit non-zero on a mismatch. Five tests cover the
failure mode, including the case where a padded file has more rows but fewer participants
than a good one.

**Blocks:** the rest of Phase 4b. Promoting notebook 09's preprocessing can proceed on
cycle G, but no cycle H result should be produced until this is reconverted.

---

## 2026-08-17 — Cohort definition promoted out of notebook 03

**Ran:** Moved the frequency-matching logic from notebook 03 into
`ambient_light_epilepsy.matching`, added `scripts/build_cohort.py` to drive it, and wrote
15 tests. Notebook 03 now imports the same functions and only explores and plots; it no
longer writes anything.

**Verified first, changed second.** Before touching the notebook, the promoted code was
run against both cycles and compared against the cohort files already in use:

```
cycle G   cases 82/82 identical   controls 276/276 identical
cycle H   cases 110/110 identical  controls 393/393 identical
```

Same participants, same order. The refactor did not alter the study population, which was
the risk worth checking — a silently different cohort would invalidate every downstream
result while still looking plausible.

**Why this one.** Cohort definition is pipeline, not exploration: it produces the files
everything downstream depends on, and it was being run by hand in a notebook whose cells
had last executed out of order. It is now one seeded command that records its own commit
hash, control ratio and seed in a provenance sidecar.

**Found while testing:** the control ratio is a **ceiling, not a target**. Strata with too
few eligible participants contribute what they have, which is why the real cohort achieves
3.37 controls per case against 4 requested. My first test asserted the ratio was met and
failed; the assertion was wrong, not the code. There are now two tests — one for the
ceiling invariant, one confirming the ratio is met exactly when the pool is deep enough —
so a genuine sampling bug stays distinguishable from thin data.

**Also:** notebook 03 re-executed cleanly top to bottom, which it could not previously be
shown to do. Two whitespace-only cells elsewhere were carrying stale outputs from deleted
code, including one reporting "Number of PWE: 110" with no code above it; cleared.

**Next:** the same promotion for the PAXMIN preprocessing in notebook 09.

---

## 2026-08-17 — Preprocessing scripts parameterised

**Ran:** No analysis. Rewrote the three R preprocessing steps and their Slurm submission
scripts so the cohort is an argument rather than an edit, and added
`scripts/lib/ale_paths.R` as the R counterpart of `paths.py`.

```
sbatch scripts/downsample_lux/downsample_lux.sh G
sbatch scripts/downsample_lux/downsample_lux.sh H 1 start
sbatch scripts/convert_xpt/convert_xpt.sh H PAXMIN
```

**Why it matters:** the cohort was previously hard-coded behind an "EDIT THIS" banner, so
running the other cohort meant editing the file, and the cycle G version was never saved.
That is why G and H preprocessing cannot currently be shown to have been identical. The
scripts can now process either cohort without modification, and echo their resolved paths
and settings into the job log.

Also: absolute RDS paths removed in favour of `ALE_PROJECT_ROOT` / `ALE_DATA_ROOT`;
`run_lux_analysis.sh` now passes arguments to the Python command-line interface and fails
loudly if the venv is missing rather than silently using the wrong one; job logs carry the
job id so reruns do not overwrite each other; `convert_xpt` skips existing parquet unless
`ALE_OVERWRITE=1`.

**Verification.** The shell scripts are syntax checked and their argument handling tested,
including the missing-argument case. The R could not be run locally — R is not installed
on the Windows workstation — so it was reviewed by inspection, which caught two bugs:
`ale_lux_dir` rejected any bin width other than 5 minutes, and `ALE_OVERWRITE=0` would
have counted as true.

**Confirmed working on BlueBEAR the same day.** `Rscript scripts/convert_xpt/convert_xpt.R H PAXMIN`
on a login node resolved:

```
Project root: /rds/.../ambient_light_epilepsy_analysis/ambient_light_epilepsy_analysis
Data root   : /rds/.../ambient_light_epilepsy_analysis/data
Already converted: PAXMIN_H.parquet
Converted: 0  skipped: 1  missing: 0
```

Both roots correct, arguments parsed, existing output skipped. Worth recording that the
repository is checked out **beside** the data on RDS rather than above it, so the data
root resolves through the `<project root>/../data` candidate rather than the first one —
the case that motivated supporting two layouts.

One cosmetic bug showed up and was fixed: `ale_check_cohort` returned its argument
visibly, so R auto-printed `[1] "H"` into every job log.

Still unexercised: the Slurm submission path, and `convert_paxlux` / `downsample_lux`
since parameterisation.

**Scope note.** The PAXLUX pipeline is now expected to be exploratory rather than
published. PAXMIN carries 1-minute light *and* activity for the same participants, which
is ample for circadian-scale analysis and allows light and activity to be compared on
identical sampling. Published results are intended to come from PAXMIN, so reprocessing
both cohorts through `convert_paxlux` and `downsample_lux` — previously the top
outstanding provenance risk — is no longer on the critical path.

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

**Next:** Phases 1 and 2 of [archive/good-practice-plan.md](archive/good-practice-plan.md).

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
