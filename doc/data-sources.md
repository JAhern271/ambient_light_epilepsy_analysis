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

A participant is classified as having epilepsy if `RXDUSE == 1` (medication taken in the
past 30 days) for any drug in `RXDDRUG` matching, case-insensitively:

    phenytoin, carbamazepine, valproic acid, divalproex sodium, phenobarbital,
    primidone, ethosuximide, levetiracetam, lamotrigine, topiramate,
    oxcarbazepine, zonisamide

Gabapentin and pregabalin are **deliberately excluded** as they are too frequently
prescribed for non-epilepsy indications to be specific.

This yields 123 PWE before the age and recording-validity filters.

> The `RXQ_RX_H` table contains a `RXDRSD1` column that fails to convert to pandas, so it
> is dropped at load time for cycle H only.

### Recording validity — `PAXHD`

Participants are included only where `PAXSTS == 1` (valid recording) and `PAXLDAY == '9'`
(full 9 days of data). Note `PAXLDAY` is compared as a **string**.

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
| `PAXTSM` | Valid seconds contributing to the minute |
| `PAXSSNMP` | Sample counter at 80 Hz; minute index is `PAXSSNMP / (60 * 80)` |
| `PAXDAYM`, `PAXDAYWM` | Day number and day of week |
| `PAXTRANM` | Transition indicator |
| `PAXQFM`, `PAXFLGSM` | Quality flag and data flags |

Non-wear is defined in notebook 09 as `PAXTSM < 45` or `PAXPREDM == 3`, and both activity
and light are masked to missing over non-wear minutes. Wear blocks shorter than 1440
minutes are discarded.

### Two sources of light data

The project has light exposure from two places, and they are not equivalent.

| | `PAXLUX` | `PAXMIN` |
|---|---|---|
| Resolution | 1 Hz, plus a derived 5-minute downsample | 1 minute |
| Activity in the same table | No | Yes |
| Non-wear handling | Not applied; metrics span the whole recording | Masked by `PAXPREDM` / `PAXTSM` |
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
| `people_with_epilepsy_{cycle}.csv` | SEQN of all identified PWE |
| `freq_match_pwe_{cycle}.csv` | SEQN of PWE entering the matched analysis |
| `freq_match_control_{cycle}.csv` | SEQN of their frequency-matched controls |

Controls are frequency matched on age band, sex, race/ethnicity, season and PIR band,
among adults (age ≥ 20) with a valid 9-day recording.

## Analysis output

`results/*/lux_*_fmatch_analysis.csv` — one row per participant, holding every derived
light metric above plus `cohort` (G/H), `epilepsy` (1 = PWE, 0 = control), and the
covariates `employed`, `depressed`, `age`, `sex`, `race`, `a_ed`, `PIR`, `NIH`, `season`
and `minutes_outdoors`.
