# -*- coding: utf-8 -*-
"""
Reader for analysis_params.toml, the single source of every analysis parameter.

Nothing in this package should contain an analysis parameter as a literal: call
`section()` instead. Three different night windows once existed across two
documents and one call site, which is the mistake this indirection prevents.

Distinct from paths.load_config(), which reads config.toml — that file holds
machine-specific data locations, and nothing in it changes a result.

    from . import params
    cohort = params.section("cohort")
    prefix = cohort["icd10_prefix"]

The file is read once and cached, since it cannot change while a process runs.
Pass an explicit `path` to read a different file; that is for tests, which need
to vary a parameter without editing the committed file.
"""

from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # Python 3.10 and earlier

from . import paths


PARAMS_FILENAME = "analysis_params.toml"

# Populated on first read. Keyed by resolved path so an override in a test does
# not poison the cached copy of the committed file.
_cache = {}


def params_file(path=None):
    """Path to the parameter file: the committed one unless told otherwise."""
    if path is not None:
        return Path(path)
    return paths.project_root() / PARAMS_FILENAME


def load(path=None):
    """Return the whole parameter file as a dict, reading it at most once."""
    resolved = params_file(path).resolve()

    if resolved not in _cache:
        if not resolved.exists():
            raise FileNotFoundError(
                f"Analysis parameters not found at {resolved}. Every analysis "
                f"parameter lives in {PARAMS_FILENAME}; it is committed, so a "
                "missing file means the repository root was resolved wrongly."
            )
        with open(resolved, "rb") as f:
            _cache[resolved] = tomllib.load(f)

    return _cache[resolved]


def section(name, path=None):
    """
    Return one top-level section, e.g. section("cohort").

    Raises rather than returning an empty dict for an unknown section: a typo
    in a section name would otherwise silently fall back to defaults, which is
    exactly the failure this module exists to prevent.
    """
    loaded = load(path)

    if name not in loaded:
        raise KeyError(
            f"No [{name}] section in {params_file(path)}. "
            f"Sections present: {sorted(loaded)}"
        )

    return loaded[name]


def clear_cache():
    """Forget any cached parameter files. For tests that rewrite one on disk."""
    _cache.clear()
