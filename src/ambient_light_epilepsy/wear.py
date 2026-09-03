# -*- coding: utf-8 -*-
"""
Non-wear masking and valid-day determination for PAXMIN (methods.md 5.1, 5.2).

This is the gate every light and activity metric sits behind. It answers three
questions, in order:

  minute       is this minute of data trustworthy?          (5.1)
  day          does this day hold enough trustworthy
               minutes to compute a metric on?              (5.2)
  participant  does this person have enough good days
               to be in the study?                          (5.2)

Promoted out of notebook 09, whose rule was a different one: it masked on
`PAXTSM < 45` or `PAXPREDM == 3`, discarded wear blocks under 1,440 minutes,
applied no quality-flag exclusion, and had no concept of a day at all. None of
those choices came from the specification, so this is a replacement rather than
a move. See doc/analysis-log.md for the date and the reasoning.

Why the timestamps are built here
---------------------------------
PAXMIN carries no clock time. The only time information in the table is
PAXSSNMP, an 80 Hz sample counter running from the first minute of the
recording, and PAXDAYM, the calendar day of wear. The clock time of a minute is

    PAXFTIME (from PAXHD) + PAXSSNMP / (60 * 80)  minutes

because the device was started when the participant left the exam centre, not
at midnight. PAXFTIME ranges from 09:11 to 21:30 across cycle H, so treating
the first minute as midnight shifts every participant's clock by a different
amount, of up to twelve and a half hours. Reconstructing PAXFTIME this way was
checked against the day-1 record count for 369 participants and matched
exactly in every case.

NHANES does not release the calendar date of a recording, so the timestamps are
anchored to ANCHOR_DATE. Only the clock time and the ordering of days carry
meaning; the date itself is synthetic and must never be reported.

Thresholds
----------
`min_valid_days` and `min_wear_hours` are **required arguments with no
defaults** throughout this module. The values in analysis_params.toml are
provisional -- the researcher reserved the choice on 2026-09-02, see
doc/implementation-status.md -- and a default would let a call site adopt an
unsettled parameter silently. Everything else comes from the [validity] section
of analysis_params.toml.
"""

import numpy as np
import pandas as pd

from . import nhanes as nhn
from . import params
from . import paths


# PAXSSNMP ticks in one minute: 80 Hz for 60 seconds
MINUTE_SAMPLES = 60 * 80

# Timestamps are anchored here because NHANES releases no calendar date. Chosen
# to be far from any daylight-saving transition so that adding minutes to it
# never skips or repeats an hour.
ANCHOR_DATE = pd.Timestamp("2013-06-03")

# The PAXMIN columns this module needs. Reading a subset matters: the full
# table is 88,223,479 rows.
MINUTE_COLUMNS = [
    "SEQN",
    "PAXDAYM",
    "PAXSSNMP",
    "PAXTSM",
    "PAXPREDM",
    "PAXMTSM",
    "PAXLXMM",
    "PAXQFM",
    "PAXFLGSM",
]

DAY_COLUMNS = ["minutes_present", "minutes_retained", "is_candidate", "is_valid"]


def _validity(overrides=None):
    """
    The [validity] parameters, with any overrides applied on top.

    Overrides exist for tests, which need to vary one rule without editing the
    committed parameter file. They are merged into the file's values rather
    than replacing them, so an override cannot accidentally drop a rule.
    """
    settings = dict(params.section("validity"))

    if overrides:
        unknown = set(overrides) - set(settings)
        if unknown:
            raise KeyError(
                f"Unknown validity parameter(s) {sorted(unknown)}. "
                f"Known: {sorted(settings)}"
            )
        settings.update(overrides)

    return settings


def clock_to_minutes(clock):
    """
    '16:30:00' -> 990, minutes past midnight.

    PAXFTIME is a fixed-width HH:MM:SS string, and single-digit hours arrive
    space-padded (' 9:11:00'), so it is stripped before parsing.
    """
    hours, minutes = clock.strip().split(":")[:2]
    return int(hours) * 60 + int(minutes)


def header_start_time(header, seqn):
    """
    One participant's PAXFTIME, from a PAXHD frame indexed by SEQN.

    Raises rather than defaulting: without PAXFTIME there is no clock time for
    the recording and nothing sensible to fall back on. A missing header row
    means the participant should not have reached this point.
    """
    key = float(seqn)

    if key not in header.index:
        raise KeyError(
            f"SEQN {seqn} has no PAXHD row, so PAXFTIME is unknown and the "
            "clock time of its minutes cannot be reconstructed."
        )

    start = header.loc[key, "PAXFTIME"]

    if not isinstance(start, str) or not start.strip():
        raise KeyError(f"SEQN {seqn} has an empty PAXFTIME in PAXHD.")

    return start


# ---------------------------------------------------------------------------
# Minute level (methods.md 5.1)
# ---------------------------------------------------------------------------

def add_clock_times(minutes, first_time):
    """
    Add a `timestamp` column, reconstructed from PAXFTIME and PAXSSNMP.

    Raises if PAXSSNMP is out of order or off the one-minute grid. Notebook 09
    responded to a non-monotonic counter by replacing it with
    `np.arange(len(df))`, which fabricates a plausible but wrong timestamp for
    every minute in the recording. A recording that fails these checks is not
    understood well enough to analyse.
    """
    minutes = minutes.copy()

    if minutes.empty:
        minutes["timestamp"] = pd.Series(dtype="datetime64[ns]")
        return minutes

    samples = minutes["PAXSSNMP"]

    if not samples.is_monotonic_increasing:
        raise ValueError(
            "PAXSSNMP is not monotonic increasing, so the minutes are out of "
            "order or duplicated. Refusing to guess the intended ordering."
        )

    off_grid = samples % MINUTE_SAMPLES != 0
    if off_grid.any():
        raise ValueError(
            f"{int(off_grid.sum())} PAXSSNMP values are not a whole number of "
            f"minutes ({MINUTE_SAMPLES} samples). A minute summary record must "
            "start on a minute boundary."
        )

    # Minutes elapsed since the recording began, then offset by the clock time
    # of its first minute.
    elapsed = (samples / MINUTE_SAMPLES).round().astype("int64")
    start = ANCHOR_DATE + pd.Timedelta(minutes=clock_to_minutes(first_time))

    minutes["timestamp"] = start + pd.to_timedelta(elapsed, unit="m")

    return minutes


def mask_minutes(minutes, validity=None):
    """
    Apply the minute-level exclusions of methods.md 5.1.

    A minute is dropped if any of these hold:

      1. the data quality review flagged it -- PAXQFM > 0, equivalently
         PAXFLGSM holding any letter. CDC states that "values >0 indicate that
         this minute is invalid based on the QC review". The two forms were
         checked against each other on all 88,223,479 rows of PAXMIN_H and
         disagreed on none, so both are tested here as belt and braces.
      2. PAXPREDM classifies it as non-wear. Sleep (code 2) and unknown
         (code 4) are kept: masking sleep would delete every night, and 5.1
         names only non-wear.
      3. the MIMS activity value is negative, i.e. CDC's "-0.01" meaning the
         value could not be computed. CDC notes that the quality review and the
         MIMS computation were run independently, so this does not imply a
         quality flag and rules 1 and 3 are both needed.
      4. PAXTSM is below `min_valid_seconds`, i.e. the minute holds too few
         seconds of data. Not named in 5.1 -- it came from notebook 09 -- and
         it excludes nothing in cycle H that rules 1 to 3 do not already
         exclude (63 such minutes in 88 million, all of them already dropped).
         Retained pending a decision on whether 5.1 should name it.

    Adds three columns:

      retained   bool, the minute survived
      mean_lux   PAXLXMM, NaN where not retained
      activity   PAXMTSM, NaN where not retained

    Light and activity are masked jointly, from one `retained` array, so that
    no participant can contribute light from a minute excluded from their
    activity metrics (5.1).
    """
    settings = _validity(validity)
    minutes = minutes.copy()

    # 1. Quality review. PAXQFM is the count of flags; PAXFLGSM is the letters.
    if settings["exclude_quality_flagged"]:
        flag_count = pd.to_numeric(minutes["PAXQFM"], errors="coerce").fillna(0)
        flag_string = minutes["PAXFLGSM"].fillna("").astype(str).str.strip()
        flagged = (flag_count > 0) | (flag_string != "")
    else:
        flagged = pd.Series(False, index=minutes.index)

    # 2. Predicted non-wear. PAXPREDM is released as a string, but a caller who
    #    has done astype(float) passes a number, so compare numerically and
    #    accept both. A string comparison against the integer parameter would
    #    match nothing and mask no minutes at all.
    predicted = pd.to_numeric(minutes["PAXPREDM"], errors="coerce")
    non_wear = predicted == settings["nonwear_pred_code"]

    # 3. Activity uncomputable. Negative covers CDC's -0.01 sentinel, which is
    #    the only negative value the variable takes.
    activity = pd.to_numeric(minutes["PAXMTSM"], errors="coerce")
    uncomputable = (activity < 0) | (activity == settings["uncomputable_mims"])

    # 4. Too few valid seconds in the minute.
    seconds = pd.to_numeric(minutes["PAXTSM"], errors="coerce")
    too_short = seconds < settings["min_valid_seconds"]

    minutes["retained"] = ~(flagged | non_wear | uncomputable | too_short)

    minutes["mean_lux"] = pd.to_numeric(
        minutes["PAXLXMM"], errors="coerce"
    ).where(minutes["retained"])
    minutes["activity"] = activity.where(minutes["retained"])

    return minutes


# ---------------------------------------------------------------------------
# Day level (methods.md 5.2)
# ---------------------------------------------------------------------------

def assign_analytic_days(minutes, validity=None):
    """
    Label each minute with the noon-to-noon day it belongs to.

    Days run noon to noon (`day_boundary_hour`) so that a night's sleep falls
    inside one analytic day rather than being split across two -- which also
    keeps the 23:00-06:00 night light window whole.

    Shifting the timestamp back by the boundary hour and taking its date gives
    the day label directly: 16:30 on day 1 and 11:00 on day 2 both shift into
    day 1, while 12:00 on day 2 shifts into day 2.

    `drop_partial_first_last` then marks the first and last of those days as
    non-candidates. They are partial by protocol: the device starts partway
    through the first and stops partway through the last (methods.md 5.2, and
    the CDC protocol, which has the participant wear it from the exam session
    to the morning of the ninth day).

    For a complete nine-day recording this leaves exactly seven candidate days
    of 1,440 minutes each, whatever time of day the device was started. The
    alternative -- keeping every day and letting the wear threshold decide --
    was rejected because it makes the candidate-day count depend on the
    participant's appointment time: a start before 16:00 leaves a first day
    with enough coverage to pass a 20 h test and a later start does not, which
    splits cycle H nearly in half on something unrelated to how the device was
    worn. See doc/analysis-log.md.

    Adds `analytic_day` (a date) and `is_candidate` (bool).
    """
    settings = _validity(validity)
    minutes = minutes.copy()

    if minutes.empty:
        minutes["analytic_day"] = pd.Series(dtype="datetime64[ns]")
        minutes["is_candidate"] = pd.Series(dtype=bool)
        return minutes

    boundary = pd.Timedelta(hours=settings["day_boundary_hour"])
    minutes["analytic_day"] = (minutes["timestamp"] - boundary).dt.normalize()

    if settings["drop_partial_first_last"]:
        first = minutes["analytic_day"].min()
        last = minutes["analytic_day"].max()
        minutes["is_candidate"] = ~minutes["analytic_day"].isin([first, last])
    else:
        minutes["is_candidate"] = True

    return minutes


def prepare_minutes(minutes, first_time, validity=None):
    """
    The whole minute-level chain: clock times, masking, then day labels.

    This is what every downstream metric should consume. The result carries
    `timestamp` and `mean_lux`, which is the frame shape lux_metrics expects,
    so those functions apply to PAXMIN light unchanged.
    """
    prepared = add_clock_times(minutes, first_time)
    prepared = mask_minutes(prepared, validity)

    return assign_analytic_days(prepared, validity)


def summarise_days(prepared, *, min_wear_hours):
    """
    One row per analytic day, with the valid-day verdict.

    Parameters
    ----------
    prepared : DataFrame
        Output of `prepare_minutes`.
    min_wear_hours : int or float
        Hours of retained data a day needs to be valid. Required, and
        deliberately without a default: the value is unsettled
        (doc/implementation-status.md).

    Returns
    -------
    DataFrame indexed by analytic day
        minutes_present    rows in the recording for that day
        minutes_retained   of those, how many survived 5.1
        is_candidate       not dropped as a partial first or last day
        is_valid           candidate AND minutes_retained >= 60 * min_wear_hours

    A day absent from the recording contributes no row, and rows missing from
    the middle of a day simply lower `minutes_present`. Nothing is zero-filled:
    a fabricated minute of zero lux and zero movement is indistinguishable from
    a genuine one spent asleep in the dark.
    """
    if prepared.empty:
        return pd.DataFrame(
            {
                "minutes_present": pd.Series(dtype="int64"),
                "minutes_retained": pd.Series(dtype="int64"),
                "is_candidate": pd.Series(dtype=bool),
                "is_valid": pd.Series(dtype=bool),
            },
            index=pd.Index([], dtype="datetime64[ns]", name="analytic_day"),
        )

    grouped = prepared.groupby("analytic_day")

    days = pd.DataFrame({
        "minutes_present": grouped.size(),
        "minutes_retained": grouped["retained"].sum().astype("int64"),
        # is_candidate is constant within a day, so any() reads it back
        "is_candidate": grouped["is_candidate"].any(),
    })

    required_minutes = 60 * min_wear_hours
    days["is_valid"] = days["is_candidate"] & (
        days["minutes_retained"] >= required_minutes
    )

    days.index.name = "analytic_day"

    return days[DAY_COLUMNS]


# ---------------------------------------------------------------------------
# Participant level (methods.md 5.2)
# ---------------------------------------------------------------------------

def summarise_participant(days, *, min_valid_days):
    """
    Collapse a day table to one verdict on the participant.

    Parameters
    ----------
    days : DataFrame
        Output of `summarise_days`.
    min_valid_days : int
        Valid days required for inclusion. Required, and deliberately without
        a default (doc/implementation-status.md).

    Returns
    -------
    Series
        n_days_recorded, n_candidate_days, n_valid_days, minutes_retained,
        meets_criterion.

    This replaces the `PAXLDAY == '9'` rule rather than amending it. That rule
    asked the header whether the device was worn to the ninth day; this asks
    the data whether enough days hold enough wear. The swap runs in both
    directions: it admits participants who stopped early but still have enough
    good days, and excludes participants who recorded all nine days but wore
    the device too little on most of them.
    """
    return pd.Series({
        "n_days_recorded": int(len(days)),
        "n_candidate_days": int(days["is_candidate"].sum()) if len(days) else 0,
        "n_valid_days": int(days["is_valid"].sum()) if len(days) else 0,
        "minutes_retained": int(days["minutes_retained"].sum()) if len(days) else 0,
        "meets_criterion": bool(
            (int(days["is_valid"].sum()) if len(days) else 0) >= min_valid_days
        ),
    })


def assess_participant(minutes, first_time, *, min_valid_days, min_wear_hours,
                       validity=None):
    """
    Run the whole chain for one participant and return (days, summary).

    Convenience wrapper, so a caller with one participant's PAXMIN rows and
    their PAXFTIME does not have to remember the order of the steps.
    """
    prepared = prepare_minutes(minutes, first_time, validity)
    days = summarise_days(prepared, min_wear_hours=min_wear_hours)

    return days, summarise_participant(days, min_valid_days=min_valid_days)


def header_only_validity(header):
    """
    The superseded rule: PAXSTS == 1 and PAXLDAY == '9'.

    Retained so the existing cohort files and everything in results/ stay
    explicable and reproducible. It is **not** the study's validity rule --
    methods.md 5.2 is, and `valid_recordings` implements it. Note PAXLDAY is
    released as a string, so it is compared as one.

    Returns the SEQN index of participants the old rule admitted.
    """
    admitted = header[(header["PAXSTS"] == 1) & (header["PAXLDAY"] == "9")]

    return admitted.index


# ---------------------------------------------------------------------------
# Reading the real table
# ---------------------------------------------------------------------------

VALIDITY_FILENAME = "valid_recordings_{cycle}.csv"


def load_header(cycle, base_path=None):
    """PAXHD for one cycle, indexed by SEQN."""
    return nhn.load_PAXHD(cycle, base_path)


def save_validity(table, cycle, base_path=None):
    """
    Write a participant-level validity table to the processed directory.

    Deciding validity means reading the whole PAXMIN table, so the result is
    written once and read back by everything downstream rather than recomputed
    per caller. `scripts/build_validity.py` is what produces it, and records
    the thresholds used in a provenance sidecar alongside.
    """
    directory = paths.processed_dir(cycle, base_path, create=True)
    path = directory / VALIDITY_FILENAME.format(cycle=cycle)

    table.to_csv(path)

    return path


def load_validity(cycle, base_path=None):
    """
    Read the validity table back, indexed by SEQN.

    Raises with instructions rather than falling back to a rule of its own:
    the thresholds are the researcher's choice, and this file is the record of
    which ones were used.
    """
    name = VALIDITY_FILENAME.format(cycle=cycle)

    try:
        path = paths.processed_file(name, cycle, base_path)
    except FileNotFoundError as missing:
        raise FileNotFoundError(
            f"No validity table for cycle {cycle}. Build it with\n"
            f"    python scripts/build_validity.py --cohort {cycle} "
            "--min-valid-days D --min-wear-hours H\n"
            "choosing D and H deliberately -- the values in "
            "analysis_params.toml are provisional (doc/implementation-status.md)."
        ) from missing

    return pd.read_csv(path, index_col="SEQN")


def valid_seqns(table):
    """The SEQN of participants a validity table admits, for `matching`."""
    return table.index[table["meets_criterion"].astype(bool)]


def load_minutes(cycle, seqns, base_path=None):
    """
    PAXMIN rows for the given participants, in one filtered pass.

    One pass for the whole list, not one per participant: PAXMIN_H is 88
    million rows, and notebook 09's per-participant filter rescans the file
    every time.
    """
    import pyarrow.dataset as ds

    wanted = [float(s) for s in seqns]
    dataset = ds.dataset(str(paths.raw_table(cycle, "PAXMIN", base_path)))

    table = dataset.to_table(
        filter=ds.field("SEQN").isin(wanted),
        columns=MINUTE_COLUMNS,
    )

    return table.to_pandas()


def iter_participants(chunks, wanted=None):
    """
    Yield (SEQN, frame) for one complete participant at a time.

    `chunks` is an iterable of row-group frames, in file order. A participant's
    minutes can straddle a row-group boundary, so the trailing participant of
    each chunk is held back until the next one arrives; without that their last
    rows would be assessed as a separate, truncated participant, losing part
    of a day silently.

    Split out from `valid_recordings` so the buffering can be tested on
    synthetic chunks rather than only against an 88-million-row file.
    """
    carried = None

    for chunk in chunks:
        if wanted is not None:
            chunk = chunk[chunk["SEQN"].isin(wanted)]

        if carried is not None and len(carried):
            chunk = pd.concat([carried, chunk], ignore_index=True)
        carried = None

        if chunk.empty:
            continue

        # Hold back the last participant: their rows may continue in the next
        # chunk. Everyone before them is complete.
        last_seqn = chunk["SEQN"].iloc[-1]
        carried = chunk[chunk["SEQN"] == last_seqn]
        complete = chunk[chunk["SEQN"] != last_seqn]

        for seqn, one in complete.groupby("SEQN", sort=False):
            yield seqn, one

    # Whatever is left after the final chunk is complete by definition.
    if carried is not None and len(carried):
        for seqn, one in carried.groupby("SEQN", sort=False):
            yield seqn, one


def valid_recordings(cycle, *, min_valid_days, min_wear_hours, seqns=None,
                     base_path=None, validity=None):
    """
    Apply methods.md 5.2 to a whole cycle: one row per participant.

    Parameters
    ----------
    min_valid_days, min_wear_hours
        Required. See the module docstring -- both are provisional and must be
        chosen by the researcher, not defaulted here.
    seqns : iterable, optional
        Restrict to these participants. Omitted, every participant in the
        table is assessed, which reads all 88 million rows of cycle H.

    Returns
    -------
    DataFrame indexed by SEQN
        The columns of `summarise_participant`, plus PAXFTIME and PAXLDAY for
        context, and `header_only_valid` so the change from the superseded
        rule can be tabulated in both directions.

    The table is streamed a row group at a time, holding back the participant
    straddling each boundary, so memory stays flat regardless of cohort size.
    """
    import pyarrow.parquet as pq

    header = load_header(cycle, base_path)
    old_rule = set(header_only_validity(header))

    wanted = None if seqns is None else {float(s) for s in seqns}

    reader = pq.ParquetFile(str(paths.raw_table(cycle, "PAXMIN", base_path)))

    chunks = (
        reader.read_row_group(group, columns=MINUTE_COLUMNS).to_pandas()
        for group in range(reader.metadata.num_row_groups)
    )

    results = [
        _assess_one(seqn, one, header, old_rule,
                    min_valid_days, min_wear_hours, validity)
        for seqn, one in iter_participants(chunks, wanted)
    ]

    if not results:
        return pd.DataFrame(columns=[
            "PAXFTIME", "PAXLDAY", "n_days_recorded", "n_candidate_days",
            "n_valid_days", "minutes_retained", "meets_criterion",
            "header_only_valid",
        ], index=pd.Index([], name="SEQN"))

    table = pd.DataFrame(results).set_index("SEQN").sort_index()

    return table


def _assess_one(seqn, one, header, old_rule, min_valid_days, min_wear_hours,
                validity):
    """Assess one participant's rows. Helper for `valid_recordings`."""
    first_time = header_start_time(header, seqn)

    _, summary = assess_participant(
        one.sort_values("PAXSSNMP"),
        first_time,
        min_valid_days=min_valid_days,
        min_wear_hours=min_wear_hours,
        validity=validity,
    )

    record = {"SEQN": float(seqn), "PAXFTIME": first_time}
    record["PAXLDAY"] = header.loc[float(seqn), "PAXLDAY"]
    record.update(summary.to_dict())
    record["header_only_valid"] = float(seqn) in old_rule

    return record
