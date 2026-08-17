# -*- coding: utf-8 -*-
"""
Recording where results came from.

Used two ways: `describe()` at the top of a notebook, so a saved notebook
carries the state it ran against, and `git_commit()` in the driver scripts,
which stamp it into a sidecar next to every results file.
"""

import platform
import subprocess
import sys
from datetime import datetime, timezone

from . import paths


def git_commit(short=False):
    """
    Current commit hash, with a -dirty suffix if the tree has changes.

    Returns "unknown" rather than raising if git is unavailable, since a
    missing commit hash should not stop an analysis from running.
    """
    try:
        repo = paths.project_root()
        rev = ["git", "rev-parse"] + (["--short"] if short else []) + ["HEAD"]
        commit = subprocess.check_output(rev, cwd=repo, text=True).strip()
        changed = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo, text=True
        ).strip()
        return commit + ("-dirty" if changed else "")
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def package_versions(names=("pandas", "numpy", "pyarrow")):
    """Versions of the packages whose behaviour could move a result."""
    versions = {}
    for name in names:
        try:
            versions[name] = __import__(name).__version__
        except ImportError:
            versions[name] = "not installed"
    return versions


def describe():
    """
    Print the state this code is running against.

    Intended as the first cell of a notebook: run it and the saved notebook
    records which commit, machine and data produced everything below.
    """
    print(f"run at       : {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"git commit   : {git_commit(short=True)}")
    print(f"machine      : {platform.node()}")
    print(f"python       : {sys.version.split()[0]}")

    for name, version in package_versions().items():
        print(f"{name:13s}: {version}")

    print()
    paths.describe()
