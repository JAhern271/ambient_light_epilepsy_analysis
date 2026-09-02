# -*- coding: utf-8 -*-
"""
Epilepsy case ascertainment from the NHANES prescription medication file.

NHANES has no epilepsy or seizure item in the medical conditions questionnaire,
so cases are identified from prescriptions. Selection is **code-first**: take
every participant reporting a medication for a condition coded ICD-10-CM G40
(epilepsy and recurrent seizures) in RXDRSC1-3, then confirm the drug is a
recognised antiseizure medication (ASM) and blank the ones that are not. See
doc/methods.md 4.1.

**Why code-first rather than drug-first.** Selecting on a list of ASM names and
then requiring G40 is not the same as selecting on G40 and confirming the drug,
because no name list is complete. In cycle H, drugs carrying a G40 code include
lacosamide, clobazam, and clonazepam, lorazepam and diazepam used for seizure
control -- all absent from a conventional twelve-drug list. Code-first recovers
these; drug-first discards them. Conversely the name list admits large numbers
of off-label users: it has a positive predictive value of 38.9% against G40 in
cycle H, mostly topiramate for migraine and divalproex or lamotrigine for mood
disorders.

**Why cycle H only.** RXQ_RX_G (2011-2012) carries no reason-for-use variables
at all -- CDC did not release them -- so any code-first definition is
implementable in cycle H alone. Cycle G is a labelled broad-definition
replication cohort (doc/methods.md 4.5), and the functions here refuse a
code-requiring definition for it rather than quietly dropping the requirement.

Four definitions, so all of the specification's sensitivity analyses come from
one function:

    primary        G40 + ASM confirmation                    (methods.md 4.1)
    narrow         G40 + ASMs rarely used off-label          (4.1, sensitivity 2)
    broad          ASM name list, no code requirement        (4.1, sensitivity 3)
    narrow_nocode  rarely-off-label names, no code           (4.5, cycle G only)

`narrow_nocode` exists because 4.1's narrow definition requires a G40 code
while 4.5 asks for the same rarely-off-label restriction in cycle G, where no
codes exist. Those are two different definitions, so they get two names rather
than one name that means different things in different cycles.

Every parameter -- the ICD-10 prefix and all four name lists -- comes from the
[cohort] section of analysis_params.toml. None of them is a literal here.
"""

import json
import platform
import sys
import warnings
from datetime import datetime, timezone

import pandas as pd
import pyarrow.parquet as pq

from . import params
from . import paths
from . import provenance


# The three reason-for-use slots. Present in RXQ_RX_H only.
REASON_CODE_COLS = ("RXDRSC1", "RXDRSC2", "RXDRSC3")

# Everything the ascertainment needs. Selecting these by name at read time also
# sidesteps RXDRSD1, a free-text column in RXQ_RX_H that fails to convert to
# pandas; the descriptions it holds duplicate the codes, so nothing is lost.
NEEDED_COLS = ("SEQN", "RXDUSE", "RXDDRUG") + REASON_CODE_COLS

# How each definition selects. `requires_code` decides whether a G40 reason
# code is needed; `drug_list` names the analysis_params.toml key holding the
# drug names, which act as *confirmation* when a code is required and as the
# *selector* when it is not.
DEFINITIONS = {
    "primary":       {"requires_code": True,  "drug_list": "asm_confirm"},
    "narrow":        {"requires_code": True,  "drug_list": "asm_narrow"},
    "broad":         {"requires_code": False, "drug_list": "asm_broad"},
    "narrow_nocode": {"requires_code": False, "drug_list": "asm_narrow"},
}


def _normalise_names(series):
    """Lower-case and strip drug names, leaving missing values missing."""
    return series.astype("string").str.strip().str.lower()


def _normalise_codes(frame):
    """
    Clean the reason-code columns.

    NHANES stores an absent code as an empty string rather than as a missing
    value, and stores codes at three-character category level (plain 'G40', not
    'G40.909'). Empty strings become missing here so that a prefix test cannot
    match one.
    """
    cleaned = frame.astype("string").apply(lambda s: s.str.strip().str.upper())
    return cleaned.replace("", pd.NA)


def load_prescriptions(cycle, base_path=None):
    """
    Load RXQ_RX for one cycle, normalised for ascertainment.

    Returns a row-per-prescription DataFrame with `drug` (lower-case),
    `current_use` (RXDUSE == 1, i.e. taken in the past 30 days) and whichever
    of the reason-code columns the cycle actually has. Cycle G has none of
    them, which callers must handle rather than assume.
    """
    table = pq.read_table(paths.raw_table(cycle, "RXQ_RX", base_path))

    present = [c for c in NEEDED_COLS if c in table.column_names]
    rx = table.select(present).to_pandas()

    rx["drug"] = _normalise_names(rx["RXDDRUG"])
    rx["current_use"] = rx["RXDUSE"] == 1

    code_cols = [c for c in REASON_CODE_COLS if c in rx.columns]
    if code_cols:
        rx[code_cols] = _normalise_codes(rx[code_cols])

    return rx


def _rows_with_code(rx, prefix):
    """
    Boolean mask of prescriptions carrying `prefix` in any reason-code slot.

    A prefix test rather than equality because the released cycle H codes are
    truncated to the three-character category, but the ICD-10 full codes
    (G40.909 and so on) would also have to match if a later release carries
    them.
    """
    code_cols = [c for c in REASON_CODE_COLS if c in rx.columns]

    return (
        rx[code_cols]
        .apply(lambda s: s.str.startswith(prefix, na=False))
        .any(axis=1)
    )


def _confirm_drugs_reviewed(rx, coded, cohort_params):
    """
    Refuse to proceed if any drug carrying the reason code is unreviewed.

    Confirmation by an allow-list alone would silently discard every G40 drug
    missing from that list -- reintroducing the incompleteness that makes
    drug-first selection wrong, one step later in the pipeline. So every drug
    observed with the code must appear on either `asm_confirm` (kept) or
    `non_asm_blanked` (blanked after manual review). Anything else stops the
    run and names itself, so a new ASM or a new coding error forces a decision.
    """
    reviewed = set(cohort_params["asm_confirm"]) | set(cohort_params["non_asm_blanked"])

    observed = rx.loc[coded, "drug"]
    # A missing drug name is unreviewed too, and would otherwise slip through
    # the set difference below.
    unreviewed = sorted(set(observed.dropna()) - reviewed)
    if observed.isna().any():
        unreviewed.append("<missing drug name>")

    if unreviewed:
        raise ValueError(
            f"{len(unreviewed)} drug(s) carry a "
            f"{cohort_params['icd10_prefix']} reason code but have not been "
            f"reviewed as antiseizure medications: {unreviewed}.\n"
            "Ascertainment has stopped rather than guess. Review each one and "
            "add it to asm_confirm (it is an ASM) or to non_asm_blanked (it is "
            "not) in analysis_params.toml, recording the reasoning in "
            "doc/analysis-log.md."
        )


def find_cases(cycle, definition=None, base_path=None, params_path=None,
               save=True, overwrite=False):
    """
    Identify epilepsy cases in one NHANES cycle.

    Parameters
    ----------
    cycle : str
        NHANES cycle letter, "G" or "H".
    definition : str, optional
        One of DEFINITIONS. Defaults to `case_definition` in
        analysis_params.toml, which is the specification's primary definition.
    save : bool
        Write the SEQN list and a provenance sidecar to the processed
        directory. Set False to compute without touching disk.
    overwrite : bool
        Recompute and rewrite even if the output file already exists. Without
        it an existing file is left alone, since a cohort file that downstream
        results were produced from should not change silently.

    Returns
    -------
    pandas.Index
        SEQN of the identified cases, as int64, sorted and unique.
    """
    cohort_params = params.section("cohort", params_path)

    if definition is None:
        definition = cohort_params["case_definition"]

    if definition not in DEFINITIONS:
        raise ValueError(
            f"Unknown case definition {definition!r}. "
            f"Expected one of {sorted(DEFINITIONS)}; see doc/methods.md 4.1."
        )

    spec = DEFINITIONS[definition]
    rx = load_prescriptions(cycle, base_path)

    has_code_cols = any(c in rx.columns for c in REASON_CODE_COLS)
    if spec["requires_code"] and not has_code_cols:
        raise ValueError(
            f"Definition {definition!r} requires a "
            f"{cohort_params['icd10_prefix']} reason code, but RXQ_RX_{cycle} "
            "carries no reason-for-use variables -- CDC did not release them "
            "for that cycle. Only cycle H supports a code-first definition; "
            "cycle G takes 'broad' or 'narrow_nocode'. See doc/methods.md 4.1."
        )

    # A prescription qualifies only if the drug and the reason code are on the
    # same row: the code is the indication for that prescription, not for the
    # participant's whole medication list.
    selected = rx["current_use"] & rx["drug"].isin(cohort_params[spec["drug_list"]])

    if spec["requires_code"]:
        coded = rx["current_use"] & _rows_with_code(rx, cohort_params["icd10_prefix"])
        # Review every coded drug, not just the ones this definition keeps, so
        # that 'narrow' is held to the same completeness standard as 'primary'.
        _confirm_drugs_reviewed(rx, coded, cohort_params)
        selected &= coded

    cases = pd.Index(
        sorted(rx.loc[selected, "SEQN"].astype("int64").unique()),
        name="SEQN",
        dtype="int64",
    )

    if save:
        _save_cases(cases, cycle, definition, cohort_params, base_path, overwrite)

    return cases


def cases_filename(cycle, definition):
    """Name of the file holding one definition's case list."""
    return f"cases_{cycle}_{definition}.csv"


def _save_cases(cases, cycle, definition, cohort_params, base_path, overwrite):
    """
    Write the case list and a sidecar recording how it was produced.

    The sidecar matters more here than for most outputs: the filename records
    which definition was used, but not which drug names that definition stood
    for on the day it ran.
    """
    save_dir = paths.processed_dir(cycle, base_path, create=True)
    save_path = save_dir / cases_filename(cycle, definition)

    if save_path.exists() and not overwrite:
        print(f"{save_path} already exists; not overwriting. "
              "Pass overwrite=True to replace it.")
        return

    pd.Series(cases, name="SEQN").to_csv(save_path, index=False)

    spec = DEFINITIONS[definition]
    record = {
        "cycle": cycle,
        "definition": definition,
        "cases": len(cases),
        "requires_reason_code": spec["requires_code"],
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "function": "cohort.find_cases",
        "git_commit": provenance.git_commit(),
        "parameters": {
            "icd10_prefix": cohort_params["icd10_prefix"],
            spec["drug_list"]: cohort_params[spec["drug_list"]],
            "non_asm_blanked": cohort_params["non_asm_blanked"],
        },
        "data_root": str(paths.data_root(base_path)),
        "machine": platform.node(),
        "python": sys.version.split()[0],
        "packages": provenance.package_versions(),
    }

    sidecar = save_path.with_suffix(".provenance.json")
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    print(f"Wrote {save_path.name} and {sidecar.name} in {save_dir}")


def load_cases(cycle, definition, base_path=None):
    """Read back a case list written by find_cases, as an Index of int64 SEQN."""
    path = paths.processed_file(cases_filename(cycle, definition), cycle, base_path)
    seqn = pd.read_csv(path)["SEQN"]

    return pd.Index(seqn.astype("int64"), name="SEQN")


def find_people_on_asm(year, base_path=None, overwrite=False):
    """
    Deprecated. Use `find_cases(cycle, definition=...)`.

    This was the drug-first definition: a twelve-name ASM list with no reason
    code requirement, which is the 'broad' definition and is not primary. It is
    kept, and delegates to that definition, so the existing cohort files and
    the results built on them stay explicable, and it still writes the legacy
    filename and format.

    It selects the same participants as before -- verified against the cycle G
    and H files on disk -- but writes them in SEQN order rather than order of
    first appearance in RXQ_RX, so a regenerated file is not byte-identical to
    the committed one. Downstream code reads the column as a set of SEQN, so
    the order does not reach any result.
    """
    warnings.warn(
        "find_people_on_asm is the drug-first 'broad' definition, which is not "
        "the study's primary case definition (doc/methods.md 4.1). Use "
        "find_cases(cycle, definition='primary') for cycle H, or "
        "find_cases(cycle, definition='broad') to keep this behaviour "
        "explicitly.",
        DeprecationWarning,
        stacklevel=2,
    )

    cases = find_cases(year, definition="broad", base_path=base_path, save=False)

    # The legacy output path and format, unchanged.
    save_dir = paths.processed_dir(year, base_path, create=True)
    save_path = save_dir / f"people_with_epilepsy_{year}.csv"

    pwe = pd.Series(cases.values, name="SEQN")

    if save_path.exists() and not overwrite:
        print(f"CSV file already exists in {save_path}")
    else:
        pwe.to_csv(save_path)
        print(f"CSV saved in {save_path}")

    return pwe


def load_pwe_seqn(year, base_path=None):

    # Load SEQN numbers for people with epilepsy
    pwe_path = paths.processed_file(f"people_with_epilepsy_{year}.csv", year, base_path)

    return pd.read_csv(pwe_path, index_col=0)



def load_freq_matched_control_groups(year, base_path=None):

    control_path = paths.processed_file(f"freq_match_control_{year}.csv", year, base_path)
    pwe_path     = paths.processed_file(f"freq_match_pwe_{year}.csv", year, base_path)

    control_s = pd.read_csv(control_path, index_col=0)
    pwe_s = pd.read_csv(pwe_path, index_col=0)

    return control_s.values.reshape(-1).astype(int), pwe_s.values.reshape(-1).astype(int)
