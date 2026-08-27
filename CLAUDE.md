# CLAUDE.md

Ambient light exposure and rest–activity rhythms in people with epilepsy, using NHANES
accelerometry and its ambient light channel.

## Document precedence

Read in this order. Later sources never override earlier ones.

1. **`doc/methods.md`** — the specification. What the study does. If code disagrees with
   it, the code is wrong.
2. **`doc/implementation-status.md`** — the single to-do list, and the spec ↔ code gap.
   Check here before concluding something is unimplemented or broken; it is probably
   already known.
3. **`doc/data-sources.md`** — descriptive reference for NHANES tables and variables.
   Records what the code *does*, so it is not a spec and may lag it.
4. **`doc/analysis-log.md`** — append-only history. Never edit past entries; add a
   superseding one.
5. **`doc/archive/**`** — **never current.** Superseded documents, kept for provenance.
   Do not act on anything in here, and do not cite it as the study design.

Two to-do lists is the failure mode this structure exists to prevent. New tasks go in
`implementation-status.md`, not into a new plan file.

## Standing decisions

These are settled and expensive to get wrong. Do not re-derive them from the code, which
in several cases still reflects the superseded design.

- **Cycle H (2013–2014) only for the primary analysis.** `RXQ_RX_G` carries no
  reason-for-use variables, so the case definition cannot be applied to cycle G at all.
- **Case definition is code-first:** select on ICD-10 `G40` in `RXDRSC1–3`, then confirm
  the drug is an ASM. The 12-name ASM list in `cohort.py` is the **broad** definition — it
  has a PPV of 38.9% against G40 and is for cycle G replication and sensitivity analysis
  only. Never present it as the primary definition.
- **Cycle G is a labelled broad-definition replication cohort**, not part of the primary
  analysis.
- **Python for metrics, R for statistics**, with a participant-level CSV at the boundary.
  Metric computation, masking and wear detection stay in `src/ambient_light_epilepsy`;
  matching, survey design and circular statistics are R (`MatchIt`, `cobalt`, `survey`,
  `circular`). Do not port tested metric code to R.
- **Analysis parameters come from `analysis_params.toml`**, never from prose in a
  document and never from a literal at a call site. Three different night windows once
  existed across two documents and one call site; that is what this rule prevents.
- **Do not cite the 192-case results in `results/` as current findings.** That cohort is
  roughly 60% off-label ASM use. Superseded, not merely exploratory.

## Running things

Use `conda run` for Python. Direct `python.exe` invocations die silently on numpy linalg
in this environment.

```bash
conda run -n base python -m pytest
```

`conda run` cannot take a `-c` argument containing newlines — write a file instead.

Data paths resolve through `config.toml` profiles (`hpc`, `w_drive`, `local`) and
`src/ambient_light_epilepsy/paths.py`. Never hard-code a path; there are three machines
and two directory layouts. Raw NHANES data is never committed.

## Conventions

- Analysis logic lives in `src/` and is tested; notebooks are numbered and exploratory.
- Test metric code against synthetic signals with hand-derived answers before changing it.
  A wrong threshold produces plausible numbers, not an error.
- Every run writes a `.provenance.json` sidecar (git commit, timestamp, parameters,
  versions). Results go to dated directories; do not overwrite a previous run.
- Add an `analysis-log.md` entry for anything whose result you would want to explain later.
