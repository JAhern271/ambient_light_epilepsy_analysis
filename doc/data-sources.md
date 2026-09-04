# Data sources and dictionary

> **Status:** descriptive reference. This file records what the code currently reads and
> computes — not what it should. The normative source is [methods.md](methods.md); where
> the two disagree that is a bug, tracked in
> [implementation-status.md](implementation-status.md).
>
> Definitions below carry a status marker where they are known to differ from the spec.

Every NHANES table used, every variable pulled from it, and how each derived variable is
defined. Cycle codes are `G` (2011–2012) and `H` (2013–2014).

NHANES is collected by the US National Center for Health Statistics and is in the public
domain. Raw files are **not** committed to this repository.

## Raw NHANES tables

| Table | Contents | Used for |
|---|---|---|
| `DEMO` | Demographics | Age, sex, race, education, income, season |
| `RXQ_RX` | Prescription medications | Identifying people with epilepsy |
| `PAXHD` | Physical activity monitor header | Recording validity |
| `PAXLUX` | Ambient light, 1 Hz | Light metrics — exploratory, see below |
| `PAXMIN` | Minute-level light **and** activity | Intended basis for published results |
| `OCQ` | Occupation | Employment status |
| `DPQ` | Depression screener (PHQ-9) | Depression status |
| `DEQ` | Dermatology | Self-reported time outdoors |
| `MCQ` | Medical conditions | Not currently used |

## Variables

### Demographics — `DEMO`

| NHANES | Renamed | Meaning |
|---|---|---|
| `SEQN` | index | Participant identifier |
| `RIDAGEYR` | `age` | Age in years at screening |
| `RIAGENDR` | `sex` | 1 = Male, 2 = Female |
| `RIDRETH3` | `race` | Race/Hispanic origin (see below) |
| `DMDEDUC3` | `p_ed` | Education, ages 6–19. Loaded then dropped |
| `DMDEDUC2` | `a_ed` | Education, ages 20+ |
| `INDFMPIR` | `PIR` | Ratio of family income to poverty threshold |
| `DMDHHSIZ` | `NIH` | Number of people in household |
| `RIDEXMON` | `season` | Six-month exam period: 1 = Nov–Apr, 2 = May–Oct |

`RIDRETH3` is mapped to: 1 Mexican American, 2 Other Hispanic, 3 Non-Hispanic White,
4 Non-Hispanic Black, 6 Non-Hispanic Asian, 7 Other/Multiracial. Note that 5 is not used
by NHANES.

`DMDEDUC2` is mapped to: 1 <9th grade, 2 9–11th grade, 3 High school/GED,
4 Some college/AA, 5 College graduate, 7 Refused, 9 Don't know.

`RIDEXMON` is labelled Winter (1) and Summer (2) in the code. These are six-month
collection periods, not meteorological seasons.

`PIR` is banded into `<1 (Low)`, `1–4 (Middle)`, `>4 (High)` using bin edges
`[0, 1, 4, 5]`. Note this convention excludes PIR exactly 0 and treats the top band as
4–5; NHANES top-codes PIR at 5.

### Epilepsy status — `RXQ_RX`

`cohort.find_cases(cycle, definition=...)` implements four definitions. All of them
require `RXDUSE == 1` (medication taken in the past 30 days), and all take their drug
lists and the ICD-10 prefix from the `[cohort]` section of `analysis_params.toml`. See
methods.md §4.1 for why, and `cohort.py` for the mechanics.

| `definition` | Reason code required | Drug list | Cycles |
|---|---|---|---|
| `primary` | `G40` in `RXDRSC1–3` | `asm_confirm` (19 names) | H only |
| `narrow` | `G40` in `RXDRSC1–3` | `asm_narrow` (4 names) | H only |
| `broad` | none | `asm_broad` (12 names) | G and H |
| `narrow_nocode` | none | `asm_narrow` | G and H |

Selection is **code-first**: the drug list confirms, it does not select. The reason code
and the drug must be on the **same prescription row**, since the code is the indication
for that prescription and not for the participant's whole medication list.

Ascertainment **raises** if a drug carries a `G40` code and appears on neither
`asm_confirm` nor `non_asm_blanked`, so an unreviewed drug name stops the run instead of
being silently dropped.

Identified counts before the age and recording-validity filters, as of 2026-09-02:
`primary` 70 (H), `narrow` 38 (H), `broad` 157 (H) and 123 (G). These are pinned by a
test in `tests/test_cohort.py`, which skips when the raw data is unreachable.

**Notes on the released data.** Reason codes are stored at three-character ICD-10
category level — plain `G40`, never `G40.909` — and an absent code is an **empty string**,
not a missing value. Both are normalised at load time; the code requirement is a prefix
test so that a fuller code in a later release would still match. In cycle H every G40 row
happens to be current use, so the `RXDUSE` filter changes nothing for the code-first
definitions, but it is applied regardless.

`RXQ_RX_G` carries **no reason-for-use variables at all**, so `primary` and `narrow` raise
for cycle G rather than quietly dropping the requirement.

> The `RXQ_RX_H` table contains a `RXDRSD1` column that fails to convert to pandas.
> `load_prescriptions` reads only the columns ascertainment needs, which sidesteps it in
> both cycles; the free-text descriptions it holds duplicate the codes.

**Superseded.** Before 2026-09-02 the only definition was the drug-first `asm_broad`
name list with no reason-code requirement, reached through
`cohort.find_people_on_asm`. That function is retained, warns, and delegates to
`definition="broad"`; it selects the same participants as before in both cycles, verified
against the committed files. It has a positive predictive value of 38.9% against G40 in
cycle H, so it is not the primary definition and everything in `results/` derives from it.

### Recording validity — `PAXHD`

| Column | Meaning |
|---|---|
| `PAXSTS` | 1 = has at least one minute of data, 2 = none. 7,776 and 1,137 in cycle H |
| `PAXFTIME` | Clock time of the **first** minute, HH:MM:SS. 09:11 to 21:30 in cycle H |
| `PAXETLDY` | Clock time at the **end** of the last minute |
| `PAXLDAY` | Last calendar day with data, 1–9. Stored as a **string** |
| `PAXFDAY` | Day of the week the recording started |

`PAXFTIME` is required to read `PAXMIN` at all — see below.

Validity is decided by `wear.valid_recordings` per methods.md §5.2, and written to
`valid_recordings_{cycle}_{rule}.csv` by `scripts/build_validity.py` — one file per
valid-day rule, named after its thresholds.
`matching.eligible_participants` takes the resulting SEQN list as a required argument.

> **Superseded 2026-09-03.** Participants used to be included where `PAXSTS == 1` **and**
> `PAXLDAY == '9'`, read straight off the header. §5.2 replaces that rule: it counts days
> with enough retained wear instead. The old rule is retained as
> `wear.header_only_validity` so the cohort files already in `data/processed` and
> everything in `results/` stay reproducible, and `scripts/build_cohort.py --validity
> legacy` applies it. It is not the study rule.

### Light — `PAXLUX`

Distributed as one archive per participant, converted to per-participant parquet files.

| Form | Path | Columns |
|---|---|---|
| 1 Hz | `PAXLUX_{cycle}/parquet/SEQN_{n}.parquet` | `HEADER_TIMESTAMP`, `LUX` |
| 5 min | `PAXLUX_{cycle}/parquet_5min/SEQN_{n}_5min.parquet` | `timestamp`, `mean_lux` |

Timestamps are labelled UTC but represent **local clock time**; this was verified in
notebook 04 by confirming that population-level first and last light exposure cluster at
07:00–09:00 and 17:00–21:00.

The 5-minute files are produced from the 1 Hz files by `scripts/downsample_lux/`, which
bins with **centre alignment**: a timestamp marks the middle of its bin, so 06:57:30
covers 06:55:00–07:00:00. This shifts samples relative to the hour boundaries the day and
night windows use, and differs between the 5-minute and 1 Hz analyses.

See `scripts/README.md` for the full preprocessing pipeline and its known limitations.

### Light and activity — `PAXMIN`

**The intended basis for published results.** PAXMIN carries minute-level ambient light
*and* physical activity for the same participants in one table, which is ample resolution
for circadian-scale analysis and lets light and activity be compared on identical
sampling. It also supersedes PAXLUX for light exposure — see *Two sources of light data*
below.

78,126,856 rows for cycle G, roughly 7,000 participants at one row per minute over 8 days.

| Column | Meaning |
|---|---|
| `PAXLXMM` | Mean ambient light for the minute, lux |
| `PAXLXSDM` | Standard deviation of light within the minute |
| `PAXMTSM` | MIMS triaxial value — the activity measure |
| `PAXAISMM` | MIMS accelerometer value |
| `PAXMXM`, `PAXMYM`, `PAXMZM` | Per-axis MIMS values |
| `PAXPREDM` | Predicted wear status; `3` denotes non-wear |
| `PAXTSM` | Seconds of data in the minute, range 3–60. **Not used** — see below |
| `PAXSSNMP` | Sample counter at 80 Hz; minute index is `PAXSSNMP / (60 * 80)` |
| `PAXDAYM`, `PAXDAYWM` | Day number and day of week |
| `PAXTRANM` | Transition indicator: the two 30 s halves of the minute were classified differently |
| `PAXQFM` | Number of quality flags on the minute. `> 0` means CDC judged it invalid |
| `PAXFLGSM` | The flag letters themselves, concatenated: `'A'`, `'AB'`, `'ABCSUWXY'` |

`PAXPREDM` is released as a **string** (`'1'`–`'4'`), as are `PAXDAYM` and `PAXDAYWM`.
Codes are 1 wake wear, 2 sleep wear, 3 non-wear, 4 unknown.

`PAXQFM` is exactly the number of letters in `PAXFLGSM`, so `PAXQFM > 0` and
`PAXFLGSM != ''` are the same rule — verified to agree on all 88,223,479 rows of
`PAXMIN_H`.

**There is no clock time in this table.** Time of day must be reconstructed as
`PAXHD.PAXFTIME + PAXSSNMP / (60 * 80)` minutes, because the device was started when the
participant left the exam centre rather than at midnight. `PAXDAYM` gives the calendar
day, so day 1 and the last day are partial and the days between them are full 1,440-minute
days. `wear.add_clock_times` does this; nothing else should.

### Non-wear and valid days — `wear.py`

Implements methods.md §5.1 and §5.2. A minute is dropped if `PAXQFM > 0` or `PAXFLGSM`
holds a letter, or `PAXPREDM == 3`, or `PAXMTSM < 0`. Those three are the whole rule.
Light and activity are masked from one shared `retained` array, so the two channels always
derive from the same minutes. In cycle H this removes **12,302,429 of 88,223,479 minutes
(13.9%)**, almost all of it `PAXPREDM` non-wear.

`PAXPREDM` sleep (code 2) and unknown (code 4, 3.3% of minutes) are **kept** — a settled
decision, not an omission. `PAXTSM` is **not** read at all: the `min_valid_seconds = 45`
rule came from notebook 09, was never in §5.1, and excluded nothing the three real rules
do not already exclude, so the parameter was deleted on 2026-09-03. Both are pinned by
tests, so reversing either is a visible change.

Days run noon to noon, and the first and last are dropped as partial by protocol. A
complete nine-day recording therefore yields exactly 7 candidate days of 1,440 minutes,
whatever time the device was started. A day is valid with `min_wear_hours` of retained
minutes, a participant included with `min_valid_days` valid days — **both thresholds are
provisional and are required arguments, not defaults.** See
[implementation-status.md](implementation-status.md).

> **Superseded 2026-09-03.** Notebook 09 defined non-wear as `PAXTSM < 45` or
> `PAXPREDM == 3`, applied no quality-flag exclusion, discarded wear blocks shorter than
> 1,440 minutes, and had no concept of a day. None of that came from the specification.
> Its `minute_of_day` column was `PAXSSNMP % 1440`, which treats the first minute as
> midnight and so mislabels every clock time by `PAXFTIME`; the column was never consumed
> downstream, so no result is affected.

### Two sources of light data

The project has light exposure from two places, and they are not equivalent.

| | `PAXLUX` | `PAXMIN` |
|---|---|---|
| Resolution | 1 Hz, plus a derived 5-minute downsample | 1 minute |
| Activity in the same table | No | Yes |
| Non-wear handling | Not applied; metrics span the whole recording | Masked per §5.1 by `wear.py` |
| Clock time | Real timestamps in the file | Reconstructed from `PAXHD.PAXFTIME` |
| Preprocessing | Three R steps, cohort previously hard-coded | Read directly from the NHANES table |
| Status | Exploratory | Intended for publication |

The PAXLUX route came first. PAXMIN was found afterwards to carry light as well, at a
resolution that is entirely sufficient here. Because PAXMIN light is already masked for
non-wear, it also resolves the outstanding problem that PAXLUX-derived metrics are
computed over non-wear time.

Results in `results/` and the analysis in notebook 08 are all PAXLUX-derived and should be
read as exploratory.

The metric code in `lux_metrics.py` is source-agnostic: it takes a frame with `timestamp`
and `mean_lux` columns, so it applies unchanged to PAXMIN light. One caveat carries over —
`IS` resamples to hourly and so is comparable across resolutions, but `IV` is inherently
resolution dependent, so minute-level IV will not match the 5-minute or 1 Hz figures.

### Employment — `OCQ`

`employed` = 1 where `OCD150` is 1 or 2 (working at a job/business, or working at a job
but absent last week), else 0.

### Depression — `DPQ`

`phq9_total` is the sum of the nine items `DPQ010`–`DPQ090`; `depressed` = 1 where
`phq9_total >= 10`, the conventional cutoff for moderate depression. Rows with any
missing item are dropped by default.

> The sum is computed across all columns present at that point, so response codes 7
> (Refused) and 9 (Don't know) would inflate the total if not already excluded. Worth
> confirming.

### Time outdoors — `DEQ`

`minutes_outdoors` is the mean of `DED120` (minutes outdoors, workday) and `DED125`
(minutes outdoors, non-workday), after replacing the special codes 3333, 7777 and 9999
with missing.

## Derived light metrics

Computed per participant by `lux_metrics.compute_lux_summary`.

| Column | Definition |
|---|---|
| `duration_hours` | Span from first to last sample |
| `mean_lux` | Mean across the whole recording |
| `mean_daytime_lux` | Mean over hours 07:00–18:59 |
| `mean_nighttime_lux` | Mean over hours 20:00–04:59 |
| `time_above_threshold` | Fraction of epochs above 1000 lux, expressed as minutes per day |
| `M10` | Highest 10-hour rolling mean of the average 24 h profile |
| `L5` | Lowest 5-hour rolling mean of the average 24 h profile |
| `RA` | Relative amplitude, `(M10 - L5) / (M10 + L5)` |
| `m10_midpoint` | Midpoint of the M10 window, minutes from midnight |
| `l5_midpoint` | Midpoint of the L5 window, minutes from midnight |
| `IS` | Interdaily stability (Witting et al. 1990), computed on **hourly** bins |
| `IV` | Intradaily variability, from successive-difference variance at the native epoch |

`IS` is the variance of the average 24 h profile as a fraction of total variance, running
from 0 (no day-to-day reproducibility) to 1 (identical days). The recording is resampled
to hourly before computation, so the profile bins and the epochs are at the same
resolution. This follows the usual convention in the nonparametric circadian literature
and means the 5-minute and 1 Hz analyses give the same answer. The `bin_size` argument
can compute at another resolution, but values are then not comparable across resolutions.

`IV` is computed at whatever resolution the input arrives at, which is standard but means
**IV from the 5-minute analysis is not comparable with IV from the 1 Hz analysis**. Higher
sampling rates yield higher IV for the same underlying signal.

M10 and L5 are computed on the average 24-hour profile with a circular extension, so
windows crossing midnight are handled.

## Derived cohort files

Written to `data/processed/`.

| File | Contents |
|---|---|
| `valid_recordings_{cycle}_{rule}.csv` | One row per participant: the methods.md 5.2 verdict under one valid-day rule, with a `.provenance.json` sidecar. `{rule}` is `d04h20` for the primary rule, `d03h16` for Su 2022 |
| `cases_{cycle}_{definition}.csv` | SEQN of cases under one case definition, with a `.provenance.json` sidecar recording the drug lists used |
| `people_with_epilepsy_{cycle}.csv` | SEQN of all identified PWE. Legacy: the drug-first `broad` definition. Kept so existing results reproduce |
| `freq_match_pwe_{cycle}.csv` | SEQN of PWE entering the matched analysis |
| `freq_match_control_{cycle}.csv` | SEQN of their frequency-matched controls |

### `valid_recordings_{cycle}_{rule}.csv`

Produced by `scripts/build_validity.py`, read by `wear.load_validity`, and turned into an
eligible-SEQN list by `wear.valid_seqns`.

**The filename names the rule.** `{rule}` comes from `wear.rule_label(D, H)`, so
`valid_recordings_H_d04h20.csv` is 4 valid days at 20 h — the primary rule, settled
2026-09-03 — and `valid_recordings_H_d03h16.csv` is Su 2022's 3 days at 16 h. The label
is derived from the thresholds rather than typed, so a filename cannot disagree with the
rule that produced it, and a sensitivity rule can never overwrite the primary table.
Callers state which rule they want: `wear.load_validity(cycle, label)` and
`scripts/build_cohort.py --validity spec --min-valid-days 4 --min-wear-hours 20`.

Columns:

| Column | Meaning |
|---|---|
| `PAXFTIME`, `PAXLDAY` | Copied from `PAXHD` for context |
| `n_days_recorded` | Noon-to-noon days the recording touches |
| `n_candidate_days` | Of those, days not dropped as a partial first or last |
| `n_valid_days` | Of those, days with at least `min_wear_hours` of retained wear |
| `minutes_retained` | Total minutes surviving 5.1 |
| `meets_criterion` | `n_valid_days >= min_valid_days`. **This is the inclusion flag** |
| `header_only_valid` | The superseded `PAXSTS == 1 and PAXLDAY == '9'` verdict, carried so the change of rule can be tabulated in both directions |

Note `meets_criterion` is written as text and relies on pandas inferring a bool dtype on
read; an object column of `"True"`/`"False"` would make `valid_seqns` admit **everyone**
without raising, so a test pins the round trip.

Controls are frequency matched on age band, sex, race/ethnicity, season and PIR band,
among adults (age ≥ 20) with a valid recording.

## Analysis output

`results/*/lux_*_fmatch_analysis.csv` — one row per participant, holding every derived
light metric above plus `cohort` (G/H), `epilepsy` (1 = PWE, 0 = control), and the
covariates `employed`, `depressed`, `age`, `sex`, `race`, `a_ed`, `PIR`, `NIH`, `season`
and `minutes_outdoors`.
