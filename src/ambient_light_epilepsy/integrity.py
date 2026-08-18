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
