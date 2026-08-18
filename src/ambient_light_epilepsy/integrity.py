# -*- coding: utf-8 -*-
"""
Checks that a converted NHANES table actually contains what it should.

Written after PAXMIN_H was found to hold only the first 2,489 of 7,776
participants, the remainder replaced by ~60 million rows of zeros. Nothing
raised an error: the file opened, had a plausible row count, and reported no
nulls. Only the participant coverage gave it away.

These functions return findings rather than printing them, so they can be
tested and called from a script.
"""

import struct

import numpy as np
import pyarrow.parquet as pq

from . import nhanes as nhn
from . import paths

# SEQN 0 is not a real participant; it is what padding looks like here
PADDING_SEQN = 0.0


def participant_coverage(path, seqn_column="SEQN"):
    """
    Summarise which participants a parquet file actually covers.

    Reads only the SEQN column, row group by row group, so this stays cheap
    on multi-gigabyte files.
    """
    handle = pq.ParquetFile(path)

    seqns = set()
    padding_rows = 0

    for group in range(handle.metadata.num_row_groups):
        values = handle.read_row_group(group, columns=[seqn_column]).column(seqn_column).to_numpy()
        padding_rows += int(np.count_nonzero(values == PADDING_SEQN))
        seqns.update(np.unique(values).tolist())

    seqns.discard(PADDING_SEQN)
    real = np.array(sorted(seqns)) if seqns else np.array([])

    return {
        "path": str(path),
        "rows": handle.metadata.num_rows,
        "padding_rows": padding_rows,
        "participants": len(real),
        "seqn_min": float(real.min()) if real.size else None,
        "seqn_max": float(real.max()) if real.size else None,
        "seqns": seqns,
    }


def check_paxmin(year, base_path=None):
    """
    Compare a PAXMIN table against the participants PAXHD says should be in it.

    Returns a dict of findings, with `problems` listing anything wrong. An
    empty `problems` means the table looks complete.
    """
    coverage = participant_coverage(paths.raw_table(year, "PAXMIN", base_path))

    header = nhn.load_PAXHD(year, base_path)
    expected = set(float(s) for s in header[header["PAXSTS"] == 1].index)

    missing = expected - coverage["seqns"]
    problems = []

    if coverage["padding_rows"]:
        share = 100 * coverage["padding_rows"] / coverage["rows"]
        problems.append(
            f"{coverage['padding_rows']:,} padding rows with SEQN=0 ({share:.0f}% of the file)"
        )

    if missing:
        share = 100 * len(missing) / len(expected)
        problems.append(
            f"{len(missing):,} of {len(expected):,} expected participants absent ({share:.0f}%)"
        )

    return {
        "cohort": year,
        "rows": coverage["rows"],
        "padding_rows": coverage["padding_rows"],
        "participants_present": coverage["participants"],
        "participants_expected": len(expected),
        "participants_missing": len(missing),
        "seqn_min": coverage["seqn_min"],
        "seqn_max": coverage["seqn_max"],
        "expected_seqn_max": float(max(expected)) if expected else None,
        "problems": problems,
    }


def cohort_availability(year, seqns, base_path=None):
    """How many of these participants are actually present in PAXMIN."""
    coverage = participant_coverage(paths.raw_table(year, "PAXMIN", base_path))

    present = [s for s in seqns if float(s) in coverage["seqns"]]
    missing = [s for s in seqns if float(s) not in coverage["seqns"]]

    return {"present": present, "missing": missing}


# ---------------------------------------------------------------------------
# Checking the raw .xpt, before conversion
# ---------------------------------------------------------------------------
#
# PAXMIN_H arrived as a full-size 9.35 GB file whose last 6.4 GB were zeros:
# an interrupted download that had preallocated its full length. Every check on
# the converted parquet reported the same fault, but none could say whether the
# cause was the conversion or the source. Checking the .xpt settles that.


def xpt_layout(path):
    """Record length and count for an XPT file, read from its header."""
    with open(path, "rb") as handle:
        head = handle.read(20000)
        handle.seek(0, 2)
        size = handle.tell()

    marker = head.find(b"HEADER RECORD*******NAMESTR HEADER RECORD")
    if marker < 0:
        raise ValueError(f"Not an XPT file, or an unexpected layout: {path}")

    n_vars = int(head[marker + 54:marker + 58])
    start = marker + 80
    record_length = sum(
        struct.unpack(">h", head[start + v * 140 + 4:start + v * 140 + 6])[0]
        for v in range(n_vars)
    )

    data_start = head.find(b"HEADER RECORD*******OBS     HEADER RECORD") + 80

    return {
        "size": size,
        "variables": n_vars,
        "record_length": record_length,
        "data_start": data_start,
        "records": (size - data_start) // record_length,
    }


def check_xpt(path, sample_records=1000):
    """
    Look for the zero padding left by an interrupted download.

    A truncated transfer that preallocated its final size leaves a file of
    exactly the right length whose tail is zeros. That reads as a valid XPT of
    the full row count, so only the content gives it away.
    """
    layout = xpt_layout(path)
    problems = []

    with open(path, "rb") as handle:
        offset = layout["data_start"] + (layout["records"] - sample_records) * layout["record_length"]
        handle.seek(max(offset, layout["data_start"]))
        tail = handle.read(sample_records * layout["record_length"])

    zero_fraction = tail.count(0) / len(tail) if tail else 1.0

    if zero_fraction == 1.0:
        # Binary search for the last record carrying any data
        with open(path, "rb") as handle:
            low, high = 0, layout["records"]
            while low < high:
                mid = (low + high) // 2
                handle.seek(layout["data_start"] + mid * layout["record_length"])
                if any(handle.read(layout["record_length"])):
                    low = mid + 1
                else:
                    high = mid
        layout["real_records"] = low
        share = 100 * low / layout["records"]
        problems.append(
            f"file is zero-filled after record {low:,} of {layout['records']:,} "
            f"({share:.0f}% real) - the download did not complete"
        )
    else:
        layout["real_records"] = layout["records"]

    layout["tail_zero_fraction"] = zero_fraction
    layout["problems"] = problems
    layout["path"] = str(path)

    return layout
