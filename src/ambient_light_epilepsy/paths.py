# -*- coding: utf-8 -*-
"""
Path resolution for the ambient light epilepsy project.

The project runs on three machines (local PC, W: network drive, BlueBEAR HPC),
each with the data in a different place. Nothing in this package should contain
a hard-coded absolute path: call the helpers here instead.

Where the data lives is resolved in this order:

1. an explicit ``base_path`` argument passed to a function
2. the ``ALE_DATA_ROOT`` environment variable
3. the profile named by ``ALE_PROFILE``, from config.toml
4. the first profile in config.toml's ``profile_order`` that exists on disk

Machine-specific overrides that should not be committed go in config.local.toml,
which takes precedence over config.toml.

The two known directory layouts are both supported, so the same code runs
against the W: drive and against a partial local copy:

    W: / HPC                        local copy
    data/G/DEMO_G.parquet           data/G/raw_parquet/DEMO_G.parquet
    data/processed/*.csv            data/G/processed/*.csv
    data/PAXLUX_G/parquet_5min/     (not copied locally)
"""

from pathlib import Path
import os

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # Python 3.10 and earlier


DATA_ROOT_ENV = "ALE_DATA_ROOT"
ANALYSIS_ROOT_ENV = "ALE_ANALYSIS_ROOT"
PROFILE_ENV = "ALE_PROFILE"


def project_root():
    """Return the repository root (the directory holding pyproject.toml)."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError(
        "Could not locate the project root: no pyproject.toml found above "
        f"{Path(__file__).resolve()}"
    )


def load_config():
    """Read config.toml, with config.local.toml overriding it if present."""
    root = project_root()
    config = {}

    for name in ("config.toml", "config.local.toml"):
        p = root / name
        if p.exists():
            with open(p, "rb") as f:
                loaded = tomllib.load(f)

            # Merge profiles rather than replacing the whole block, so a local
            # file can override one profile without restating the others.
            profiles = config.get("profiles", {})
            profiles.update(loaded.pop("profiles", {}))
            config.update(loaded)
            if profiles:
                config["profiles"] = profiles

    return config


def _resolve(path_str):
    """Expand a config path, treating relative paths as relative to the repo."""
    p = Path(os.path.expandvars(str(path_str))).expanduser()
    return p if p.is_absolute() else (project_root() / p)


def _root_from_config(key, env_var):
    """Shared resolution logic for the data and analysis roots."""
    env_value = os.environ.get(env_var)
    if env_value:
        return _resolve(env_value)

    config = load_config()
    profiles = config.get("profiles", {})

    if not profiles:
        raise RuntimeError(
            f"No profiles defined in config.toml, and {env_var} is not set. "
            f"Set {env_var} to the {key.replace('_', ' ')}."
        )

    # An explicitly requested profile must exist, and is used as-is.
    requested = os.environ.get(PROFILE_ENV)
    if requested:
        if requested not in profiles:
            raise RuntimeError(
                f"{PROFILE_ENV}={requested!r} but config.toml defines only "
                f"{sorted(profiles)}"
            )
        return _resolve(profiles[requested][key])

    # Otherwise take the first profile that is actually present on this machine.
    order = config.get("profile_order", sorted(profiles))
    tried = []
    for name in order:
        if name not in profiles or key not in profiles[name]:
            continue
        candidate = _resolve(profiles[name][key])
        if candidate.exists():
            return candidate
        tried.append(f"  {name}: {candidate}")

    raise FileNotFoundError(
        f"Could not find a {key.replace('_', ' ')} on this machine. Tried:\n"
        + "\n".join(tried)
        + f"\nSet {env_var}, or add a profile to config.local.toml."
    )


def data_root(base_path=None):
    """Return the data root directory."""
    if base_path is not None:
        return Path(base_path)
    return _root_from_config("data_root", DATA_ROOT_ENV)


def analysis_root(base_path=None):
    """Return the directory analysis outputs are written to."""
    if base_path is not None:
        return Path(base_path)
    return _root_from_config("analysis_root", ANALYSIS_ROOT_ENV)


def _first_existing(candidates, description):
    """Return the first candidate that exists, else raise a helpful error."""
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find {description}. Tried:\n"
        + "\n".join(f"  {c}" for c in candidates)
    )


def raw_table(year, table, base_path=None):
    """
    Path to a raw NHANES table, e.g. raw_table("G", "DEMO").

    Handles both the flat layout (data/G/DEMO_G.parquet) and the local copy
    (data/G/raw_parquet/DEMO_G.parquet).
    """
    root = data_root(base_path)
    filename = f"{table}_{year}.parquet"

    return _first_existing(
        [root / year / filename, root / year / "raw_parquet" / filename],
        f"NHANES table {table} for cohort {year}",
    )


def raw_xpt(year, table, base_path=None):
    """
    Path to a raw NHANES .xpt, whether or not it has been converted yet.

    Deliberately independent of raw_table: the source must be checkable when
    the parquet is missing, which is exactly the situation after deleting a
    bad conversion.
    """
    root = data_root(base_path)
    filename = f"{table}_{year}.xpt"

    return _first_existing(
        [root / year / filename, root / year / "raw_xpt" / filename],
        f"raw XPT for {table}, cohort {year}",
    )


def processed_dir(year=None, base_path=None, create=False):
    """
    Directory holding derived cohort files (PWE lists, matched controls).

    Reads tolerate either layout; writes always go to data/processed so that
    one canonical location is used going forward.
    """
    root = data_root(base_path)
    canonical = root / "processed"

    if create:
        canonical.mkdir(parents=True, exist_ok=True)
        return canonical

    candidates = [canonical]
    if year is not None:
        candidates.append(root / year / "processed")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return canonical


def processed_file(name, year=None, base_path=None):
    """Path to a derived cohort file, e.g. processed_file("freq_match_pwe_G.csv", "G")."""
    root = data_root(base_path)
    candidates = [root / "processed" / name]
    if year is not None:
        candidates.append(root / year / "processed" / name)

    return _first_existing(candidates, f"processed file {name}")


def lux_dir(year, downsample="5min", base_path=None):
    """Directory of per-participant LUX parquet files for a cohort."""
    root = data_root(base_path)

    if downsample == "5min":
        return root / f"PAXLUX_{year}" / "parquet_5min"
    elif downsample is None:
        return root / f"PAXLUX_{year}" / "parquet"
    else:
        raise ValueError(
            f"Unknown downsample {downsample!r}; expected '5min' or None"
        )


def lux_file(seqn, year, downsample="5min", base_path=None):
    """Path to one participant's LUX parquet file."""
    directory = lux_dir(year, downsample, base_path)

    if downsample == "5min":
        return directory / f"SEQN_{int(seqn)}_5min.parquet"
    else:
        return directory / f"SEQN_{int(seqn)}.parquet"


def describe():
    """Print the resolved paths. Useful as a first cell in a notebook."""
    print(f"project root : {project_root()}")
    try:
        print(f"data root    : {data_root()}")
    except (FileNotFoundError, RuntimeError) as e:
        print(f"data root    : UNRESOLVED ({e})")
    try:
        print(f"analysis root: {analysis_root()}")
    except (FileNotFoundError, RuntimeError) as e:
        print(f"analysis root: UNRESOLVED ({e})")
